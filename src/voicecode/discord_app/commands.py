"""Slash commands.

Every command re-checks scope through `interaction_gate` before doing anything. That
is redundant with guild-scoped registration, and deliberately so: registration scope is
a Discord-side convenience that says which commands are *offered*, not which are
*honoured*.

/approve and /deny are the only paths that can answer a permission prompt. Nothing
spoken ever resolves one.
"""

from __future__ import annotations

import logging

import discord
from discord import app_commands

from ..bridge.base import PermissionDecision
from ..config import WakeMode, describe_scope
from .scoping import interaction_gate, log_refusal

log = logging.getLogger(__name__)


def register_commands(bot) -> None:  # noqa: C901 - a flat command table reads better
    tree = bot.tree

    async def guard(interaction: discord.Interaction) -> bool:
        """Gate 3. Returns True if the caller may proceed."""
        decision = interaction_gate(
            bot.config.snapshot,
            getattr(interaction.guild, "id", None),
            getattr(interaction.user, "id", None),
        )
        if decision:
            return True
        log_refusal(
            "interaction",
            decision,
            guild=getattr(interaction.guild, "id", None),
            user=getattr(interaction.user, "id", None),
            command=getattr(interaction.command, "name", None),
        )
        await interaction.response.send_message(f"Refused: {decision.reason}", ephemeral=True)
        return False

    # -- connection ------------------------------------------------------------

    @tree.command(name="join", description="Join your voice channel and start listening")
    @app_commands.describe(channel="Voice channel to join (defaults to yours)")
    async def join(
        interaction: discord.Interaction, channel: discord.VoiceChannel | None = None
    ) -> None:
        if not await guard(interaction):
            return
        target = channel or getattr(getattr(interaction.user, "voice", None), "channel", None)
        await interaction.response.defer(ephemeral=True)
        ok, message = await bot.join_if_allowed(
            getattr(interaction.guild, "id", None), target
        )
        await interaction.followup.send(
            message if ok else f"Refused: {message}", ephemeral=True
        )

    @tree.command(name="leave", description="Leave the voice channel")
    async def leave(interaction: discord.Interaction) -> None:
        if not await guard(interaction):
            return
        left = await bot.leave_guild_voice(interaction.guild.id, "requested")
        await interaction.response.send_message(
            "Left the voice channel." if left else "Not connected.", ephemeral=True
        )

    # -- audio -----------------------------------------------------------------

    @tree.command(name="mute", description="Stop or resume transcribing without leaving")
    @app_commands.describe(muted="True to stop listening, False to resume")
    async def mute(interaction: discord.Interaction, muted: bool = True) -> None:
        if not await guard(interaction):
            return
        session = bot.session_for(interaction.guild.id)
        sink = getattr(session, "sink", None) if session else None
        if sink is None:
            await interaction.response.send_message("Not connected.", ephemeral=True)
            return
        sink.set_muted(muted)
        await interaction.response.send_message(
            "Muted — audio is dropped at the sink." if muted else "Listening again.",
            ephemeral=True,
        )

    @tree.command(name="voice", description="Change the Kokoro voice")
    @app_commands.describe(name="Kokoro voice name, e.g. af_heart")
    async def voice(interaction: discord.Interaction, name: str) -> None:
        if not await guard(interaction):
            return
        bot.tts.set_voice(name)
        await interaction.response.send_message(f"Voice set to `{name}`.", ephemeral=True)

    @tree.command(name="mode", description="Set the wake gate")
    @app_commands.describe(gate="always, wakeword, or ptt")
    @app_commands.choices(
        gate=[
            app_commands.Choice(name="always", value="always"),
            app_commands.Choice(name="wakeword", value="wakeword"),
            app_commands.Choice(name="ptt", value="ptt"),
        ]
    )
    async def mode(interaction: discord.Interaction, gate: str) -> None:
        if not await guard(interaction):
            return
        bot.config.settings.wake_mode = WakeMode(gate)
        extra = ""
        if gate == "wakeword":
            extra = f" Wake word: “{bot.config.settings.wake_word}”."
        elif gate == "ptt":
            extra = " Use `/ptt` to open and close the mic."
        await interaction.response.send_message(
            f"Wake mode set to `{gate}`.{extra}", ephemeral=True
        )

    @tree.command(name="ptt", description="Open or close push-to-talk")
    @app_commands.describe(open="True to start accepting speech, False to stop")
    async def ptt(interaction: discord.Interaction, open: bool = True) -> None:
        if not await guard(interaction):
            return
        session = bot.session_for(interaction.guild.id)
        if session is None:
            await interaction.response.send_message("Not connected.", ephemeral=True)
            return
        session.set_ptt(open)
        await interaction.response.send_message(
            "Push-to-talk open." if open else "Push-to-talk closed.", ephemeral=True
        )

    @tree.command(name="stop", description="Stop in-flight playback")
    async def stop(interaction: discord.Interaction) -> None:
        if not await guard(interaction):
            return
        session = bot.session_for(interaction.guild.id)
        if session is None:
            await interaction.response.send_message("Not connected.", ephemeral=True)
            return
        dropped = session.stop_playback()
        await interaction.response.send_message(
            f"Playback stopped ({dropped} frame(s) dropped).", ephemeral=True
        )

    # -- Claude Code -----------------------------------------------------------

    @tree.command(name="interrupt", description="Interrupt the in-flight Claude Code turn")
    async def interrupt(interaction: discord.Interaction) -> None:
        if not await guard(interaction):
            return
        await interaction.response.defer(ephemeral=True)
        await bot.bridge.interrupt()
        session = bot.session_for(interaction.guild.id)
        if session is not None:
            session.stop_playback()
        await interaction.followup.send("Interrupt sent to Claude Code.", ephemeral=True)

    @tree.command(name="approve", description="Approve the pending Claude Code permission prompt")
    async def approve(interaction: discord.Interaction) -> None:
        if not await guard(interaction):
            return
        session = bot.session_for(interaction.guild.id)
        if session is None:
            await interaction.response.send_message("Not connected.", ephemeral=True)
            return
        pending = session.pending_permission
        if pending is None:
            await interaction.response.send_message(
                "No permission prompt is pending.", ephemeral=True
            )
            return
        await interaction.response.defer(ephemeral=True)
        log.warning(
            "permission APPROVED by %s (%s) for tool %r",
            interaction.user, getattr(interaction.user, "id", "?"), pending.tool,
        )
        try:
            message = await session.resolve_permission(PermissionDecision.APPROVE)
        except Exception as exc:
            message = f"Could not approve: {exc}"
        await interaction.followup.send(message, ephemeral=True)

    @tree.command(name="deny", description="Deny the pending Claude Code permission prompt")
    async def deny(interaction: discord.Interaction) -> None:
        if not await guard(interaction):
            return
        session = bot.session_for(interaction.guild.id)
        if session is None or session.pending_permission is None:
            await interaction.response.send_message(
                "No permission prompt is pending.", ephemeral=True
            )
            return
        await interaction.response.defer(ephemeral=True)
        try:
            message = await session.resolve_permission(PermissionDecision.DENY)
        except Exception as exc:
            message = f"Could not deny: {exc}"
        await interaction.followup.send(message, ephemeral=True)

    # -- ops -------------------------------------------------------------------

    @tree.command(name="reload", description="Reload allowlists from .env without restarting")
    async def reload(interaction: discord.Interaction) -> None:
        if not await guard(interaction):
            return
        await interaction.response.defer(ephemeral=True)
        summary = await bot.reload_config()
        await interaction.followup.send(f"Config reloaded.\n{summary}", ephemeral=True)

    @tree.command(name="status", description="Auth method, channel scope, and GPU state")
    async def status(interaction: discord.Interaction) -> None:
        if not await guard(interaction):
            return
        await interaction.response.defer(ephemeral=True)

        health = await bot.bridge.health()
        session = bot.session_for(interaction.guild.id)
        lines = [
            "**Claude Code**",
            health.describe(),
            "",
            "**Scope**",
            describe_scope(bot.config.snapshot),
            "",
            "**Models**",
            bot.asr.describe(),
            bot.tts.describe(),
            _gpu_line(),
        ]
        if session is not None:
            lines += [
                "",
                "**Voice**",
                f"channel: {session.channel_id}",
                f"wake mode: {bot.config.settings.wake_mode.value}",
                f"playback active: {session.playback.active}",
                f"rate limited: {session.rate_limited}",
            ]
            dave = getattr(session.sink, "decryptor", None)
            if dave is not None:
                lines.append(f"DAVE: {dave.status.describe()}")
            if session.pending_permission is not None:
                lines.append(
                    f"⛔ awaiting permission for `{session.pending_permission.tool}`"
                )
        else:
            lines += ["", "**Voice**", "not connected"]

        await interaction.followup.send("\n".join(lines)[:1900], ephemeral=True)


def _gpu_line() -> str:
    try:
        import torch
    except ImportError:
        return "GPU: torch not installed"
    if not torch.cuda.is_available():
        return "GPU: CUDA not available (running on CPU)"
    try:
        free, total = torch.cuda.mem_get_info()
        name = torch.cuda.get_device_name(0)
        return f"GPU: {name}, {free / 1e9:.1f} / {total / 1e9:.1f} GB free"
    except Exception as exc:  # pragma: no cover
        return f"GPU: CUDA available, could not read memory ({exc})"
