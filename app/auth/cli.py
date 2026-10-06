"""Command-line utility for generating password hashes and credentials for deployment."""

from __future__ import annotations

import argparse
import sys

from app.auth.security import hash_password


def main() -> None:
    parser = argparse.ArgumentParser(description="Virtual TES Authentication Management CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    hash_cmd = subparsers.add_parser("hash", help="Generate a secure PBKDF2-SHA256 password hash")
    hash_cmd.add_argument("password", type=str, help="Plaintext password to hash")

    args = parser.parse_args()
    if args.command == "hash":
        hashed = hash_password(args.password)
        print("\nGenerated PBKDF2-SHA256 Hash:")
        print("--------------------------------------------------------------------------------")
        print(hashed)
        print("--------------------------------------------------------------------------------")
        print("Use this value for DASHBOARD_PASSWORD_HASH or VIEWER_PASSWORD_HASH in Railway Variables.\n")


if __name__ == "__main__":
    main()
