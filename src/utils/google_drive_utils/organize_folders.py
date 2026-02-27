
from src.utils.google_drive_utils.authenticate_google_drive import GoogleDriveAuth


class FolderOrganizer:
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
                    fields="nextPageToken, files(id, name, mimeType, parents)",
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
    def create_folder(service, parent_folder_id: str, folder_name: str) -> str:
        file_metadata = {
            "name": folder_name,
            "mimeType": "application/vnd.google-apps.folder",
            "parents": [parent_folder_id],
        }
        folder = service.files().create(body=file_metadata, fields="id").execute()
        return folder.get("id")

    @staticmethod
    def move_file(service, file_id: str, new_folder_id: str) -> None:
        file = service.files().get(fileId=file_id, fields="parents").execute()
        previous_parents = ",".join(file.get("parents"))
        service.files().update(
            fileId=file_id,
            addParents=new_folder_id,
            removeParents=previous_parents,
            fields="id, parents",
        ).execute()

    @staticmethod
    def copy_file(service, file_id: str, new_folder_id: str, new_name: str | None = None) -> str:
        file_metadata: dict = {"parents": [new_folder_id]}
        if new_name:
            file_metadata["name"] = new_name
        copied_file = service.files().copy(fileId=file_id, body=file_metadata).execute()
        return copied_file.get("id")

    @staticmethod
    def extract_image_number(file_name: str) -> int | None:
        try:
            return int(file_name.split("_")[1].split(".")[0])
        except (IndexError, ValueError):
            return None

    @staticmethod
    def copy_images_in_range(
        service, items: list[dict], start_num: int, end_num: int, folder_id: str, flower_id: str
    ) -> None:
        for item in items:
            image_number = FolderOrganizer.extract_image_number(item["name"])
            if image_number is not None and start_num <= image_number <= end_num:
                FolderOrganizer.copy_file(service, item["id"], folder_id)

    @staticmethod
    def organize_images(
        service,
        source_folder_id: str,
        dest_folder_id: str,
        flower_ids: list[int],
        flower_id_images: list[str],
    ) -> None:
        items_sorted = sorted(
            FolderOrganizer.list_files_in_folder(service, source_folder_id),
            key=lambda x: x["name"],
        )
        flower_folders: dict = {}
        image_numbers = [FolderOrganizer.extract_image_number(name) for name in flower_id_images]

        for index, flower_id in enumerate(flower_ids):
            start_num = image_numbers[index]
            end_num = (
                image_numbers[index + 1] - 1
                if index + 1 < len(image_numbers)
                else FolderOrganizer.extract_image_number(items_sorted[-1]["name"])
            )

            flower_folder_id = flower_folders.get(flower_id)
            if not flower_folder_id:
                flower_folder_id = FolderOrganizer.create_folder(service, dest_folder_id, str(flower_id))
                flower_folders[flower_id] = flower_folder_id

            FolderOrganizer.copy_images_in_range(
                service, items_sorted, start_num, end_num, flower_folder_id, str(flower_id)
            )

    @staticmethod
    def create_main_and_subfolders(
        service, parent_folder_id: str, main_folder_name: str
    ) -> dict:
        file_metadata = {
            "name": main_folder_name,
            "mimeType": "application/vnd.google-apps.folder",
            "parents": [parent_folder_id],
        }
        main_folder = service.files().create(body=file_metadata, fields="id").execute()
        main_folder_id = main_folder["id"]

        subfolder_names = ["lab_unorganized", "lab", "greenhouse_unorganized", "greenhouse"]
        subfolder_ids: dict = {}

        for subfolder_name in subfolder_names:
            subfolder_metadata = {
                "name": subfolder_name,
                "mimeType": "application/vnd.google-apps.folder",
                "parents": [main_folder_id],
            }
            subfolder = service.files().create(body=subfolder_metadata, fields="id").execute()
            subfolder_ids[subfolder_name] = subfolder["id"]

        return {"main_folder_id": main_folder_id, "subfolders": subfolder_ids}


if __name__ == "__main__":
    service = GoogleDriveAuth.get_drive_service()

    source_folder_id = "1jXGDvVvqlLQQSHUFnl3WSAB02y-q2MBz"
    dest_folder_id = "1Kfgw-L-R-aDS2rNi20Kfh8UqY1Vtk1_1"

    flower_ids = list(range(136, 151))
    image_ids = [4576, 4604, 4630, 4658, 4684, 4719, 4747, 4783, 4813, 4837, 4867, 4894, 4920, 4951, 4977]
    flower_id_images = [f"IMG_{image_id}.JPG" for image_id in image_ids]

    FolderOrganizer.organize_images(service, source_folder_id, dest_folder_id, flower_ids, flower_id_images)
