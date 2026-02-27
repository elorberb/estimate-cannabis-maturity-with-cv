from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

SCRIPT_DIR = Path(__file__).parent
CREDS_DIR = SCRIPT_DIR / "creds"
CLIENT_SECRET_FILE = CREDS_DIR / "client_secret.json"
TOKEN_PATH = CREDS_DIR / "token.json"
SCOPES = ["https://www.googleapis.com/auth/drive"]


class GoogleDriveAuth:
    @staticmethod
    def get_drive_service():
        creds = None

        if TOKEN_PATH.exists():
            creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), SCOPES)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                if not CLIENT_SECRET_FILE.exists():
                    raise FileNotFoundError(
                        f"Client secret file not found at {CLIENT_SECRET_FILE}\n"
                        "Please download your OAuth2 credentials from Google Cloud Console "
                        "and save them as 'client_secret.json' in the 'creds' folder."
                    )
                flow = InstalledAppFlow.from_client_secrets_file(str(CLIENT_SECRET_FILE), SCOPES)
                creds = flow.run_local_server(port=0)

            CREDS_DIR.mkdir(exist_ok=True)
            with open(TOKEN_PATH, "w") as token_file:
                token_file.write(creds.to_json())

        return build("drive", "v3", credentials=creds)


if __name__ == "__main__":
    service = GoogleDriveAuth.get_drive_service()
    print("Authentication successful! Token saved to creds/token.json")
