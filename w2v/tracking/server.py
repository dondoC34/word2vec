"""
MLflow tracking server lifecycle management module.

This module handles:
- Probing whether an MLflow tracking server instance is actively listening on a network port.
- Spawning the official MLflow tracking server and Web UI dashboard as a managed process.
"""

from __future__ import annotations

import logging
import socket
import subprocess
import sys

logger = logging.getLogger("MLflowServer")


def check_mlflow_server_online(host: str = "127.0.0.1", port: int = 5000, timeout: float = 0.5) -> bool:
    """
    Checks whether an MLflow server instance is actively listening on the target host and port.

    Args:
        host: Hostname or IP address to probe.
        port: TCP port number to connect to.
        timeout: Socket connection timeout in seconds.

    Returns:
        bool: True if connection succeeded, False if offline or timed out.
    """
    try:
        with socket.create_connection((host, int(port)), timeout=timeout):
            return True
    except (OSError, ValueError):
        return False


def start_mlflow_server(
    host: str = "127.0.0.1",
    port: int = 5000,
    backend_store_uri: str = "sqlite:///mlflow.db",
    default_artifact_root: str = "./mlartifacts",
) -> int:
    """
    Launches the MLflow tracking server and web UI as a subprocess.

    Args:
        host: Host IP address to bind the server to (default: 127.0.0.1).
        port: Port number to bind the server to (default: 5000).
        backend_store_uri: Database storage URI (e.g. sqlite:///mlflow.db).
        default_artifact_root: File directory path where artifacts are stored.

    Returns:
        int: Process exit code (0 on clean termination or keyboard interrupt).
    """
    cmd = [
        sys.executable,
        "-m",
        "mlflow",
        "server",
        "--host",
        str(host),
        "--port",
        str(port),
        "--backend-store-uri",
        str(backend_store_uri),
        "--default-artifact-root",
        str(default_artifact_root),
    ]
    logger.info("=" * 70)
    logger.info("Starting MLflow Server...")
    logger.info("Backend SQLite URI: %s", backend_store_uri)
    logger.info("Default Artifact Root: %s", default_artifact_root)
    logger.info("MLflow Web Dashboard: http://%s:%s", host, port)
    logger.info("=" * 70)
    print(f"\n[w2v] MLflow UI running at: http://{host}:{port}")
    print("[w2v] Press Ctrl+C to terminate the server.\n")
    try:
        return subprocess.call(cmd)
    except KeyboardInterrupt:
        logger.info("MLflow server stopped by user.")
        return 0
