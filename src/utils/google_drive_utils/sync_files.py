import io
import os

from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload

from src.utils.google_drive_utils.authenticate_google_drive import GoogleDriveAuth


class DriveSync:
    @staticmethod
    def list_files_in_folder(service, folder_id: str) -> list[dict]:
        items = []
        query = f"'{folder_id}' in parents and trashed=false"
        page_token = None

        while True:
            response = (
                service.files()
                .list(
                    q=query,
                    spaces="drive",
                    fields="nextPageToken, files(id, name, mimeType)",
                    pageToken=page_token,
                )
                .execute()
            )
            items.extend(response.get("files", []))
            page_token = response.get("nextPageToken", None)
            if page_token is None:
                break
        return items

    @staticmethod
    def download_file(service, file_id: str, file_path: str, overwrite: bool = False) -> None:
        if not overwrite and os.path.exists(file_path):
            return
        request = service.files().get_media(fileId=file_id)
        fh = io.FileIO(file_path, "wb")
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()

    @staticmethod
    def download_folder(service, folder_id: str, local_path: str, overwrite: bool = False) -> None:
        items = DriveSync.list_files_in_folder(service, folder_id)
        for item in items:
            item_name = item["name"]
            item_id = item["id"]
            item_mime_type = item["mimeType"]
            local_item_path = os.path.join(local_path, item_name)

            if item_mime_type == "application/vnd.google-apps.folder":
                if not os.path.exists(local_item_path):
                    os.makedirs(local_item_path)
                DriveSync.download_folder(service, item_id, local_item_path, overwrite)
            elif not item_mime_type.startswith("application/vnd.google-apps."):
                DriveSync.download_file(service, item_id, local_item_path, overwrite)

    @staticmethod
    def delete_empty_folders(path: str) -> None:
        if not os.path.isdir(path):
            return

        for root, dirs, _ in os.walk(path, topdown=False):
            for dir_name in dirs:
                dir_path = os.path.join(root, dir_name)
                try:
                    if not os.listdir(dir_path):
                        os.rmdir(dir_path)
                except OSError:
                    pass

        try:
            if not os.listdir(path):
                os.rmdir(path)
        except OSError:
            pass

    @staticmethod
    def upload_file(service, file_path: str, folder_id: str) -> None:
        file_name = os.path.basename(file_path)
        query = f"name='{file_name}' and '{folder_id}' in parents and trashed=false"
        response = service.files().list(q=query, spaces="drive", fields="files(id, name)").execute()
        if response.get("files", []):
            return

        media = MediaFileUpload(file_path, resumable=True)
        file_metadata = {"name": file_name, "parents": [folder_id]}
        service.files().create(body=file_metadata, media_body=media, fields="id").execute()

    @staticmethod
    def create_folder(service, folder_name: str, parent_folder_id: str) -> str:
        query = f"name='{folder_name}' and '{parent_folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
        response = service.files().list(q=query, spaces="drive", fields="files(id, name)").execute()
        if response.get("files", []):
            return response["files"][0]["id"]

        folder_metadata = {
            "name": folder_name,
            "mimeType": "application/vnd.google-apps.folder",
            "parents": [parent_folder_id],
        }
        folder = service.files().create(body=folder_metadata, fields="id").execute()
        return folder.get("id")

    @staticmethod
    def upload_folder_dfs(service, local_path: str, parent_folder_id: str) -> None:
        stack = [(local_path, parent_folder_id)]

        while stack:
            current_local_path, current_parent_id = stack.pop()

            for entry in os.listdir(current_local_path):
                full_path = os.path.join(current_local_path, entry)
                if os.path.isdir(full_path):
                    new_folder_id = DriveSync.create_folder(service, entry, current_parent_id)
                    stack.append((full_path, new_folder_id))
                else:
                    DriveSync.upload_file(service, full_path, current_parent_id)

    @staticmethod
    def check_storage_quota(service) -> None:
        about = service.about().get(fields="storageQuota").execute()
        quota = about.get("storageQuota", {})
        usage = int(quota.get("usage", 0))
        limit = int(quota.get("limit", 0))
        print(f"Storage used: {usage / (1024**3):.2f} GB")
        print(f"Storage limit: {limit / (1024**3):.2f} GB")
        if limit > 0:
            print(f"Storage usage percentage: {usage / limit * 100:.2f}%")
        else:
            print("Unlimited storage")


if __name__ == "__main__":
    service = GoogleDriveAuth.get_drive_service()

    folder_id = "1Kfgw-L-R-aDS2rNi20Kfh8UqY1Vtk1_1"
    local_download_path = "/sise/shanigu-group/etaylor/assessing_cannabis_exp/experiment_2/images/day_9_2025_01_16/lab"

    if not os.path.exists(local_download_path):
        os.makedirs(local_download_path)

    DriveSync.download_folder(service, folder_id, local_download_path, False)
