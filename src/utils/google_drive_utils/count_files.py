import csv
import sys
from pathlib import Path

from src.utils.google_drive_utils.authenticate_google_drive import GoogleDriveAuth

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".tif",
    ".webp", ".heic", ".raw", ".dng", ".cr2", ".nef",
}


class FileCounter:
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
    def is_image(filename: str, mime_type: str) -> bool:
        ext = Path(filename).suffix.lower()
        return ext in IMAGE_EXTENSIONS or mime_type.startswith("image/")

    @staticmethod
    def count_images_per_folder(
        service,
        folder_id: str,
        folder_name: str = "root",
        path: str = "",
        results: list | None = None,
    ) -> tuple[int, list[dict]]:
        if results is None:
            results = []

        current_path = f"{path}/{folder_name}" if path else folder_name
        items = FileCounter.list_files_in_folder(service, folder_id)

        direct_image_count = 0
        subfolders = []

        for item in items:
            mime_type = item["mimeType"]
            if mime_type == "application/vnd.google-apps.folder":
                subfolders.append(item)
            elif FileCounter.is_image(item["name"], mime_type):
                direct_image_count += 1

        subfolder_image_count = 0
        for subfolder in subfolders:
            sub_total, _ = FileCounter.count_images_per_folder(
                service, subfolder["id"], subfolder["name"], current_path, results
            )
            subfolder_image_count += sub_total

        total_images = direct_image_count + subfolder_image_count

        results.append({
            "path": current_path,
            "folder_name": folder_name,
            "direct_images": direct_image_count,
            "total_images": total_images,
            "depth": current_path.count("/"),
        })

        return total_images, results

    @staticmethod
    def print_folder_image_counts(results: list[dict]) -> None:
        results_sorted = sorted(results, key=lambda x: x["path"])

        print(f"\n{'=' * 70}")
        print("IMAGE COUNTS PER FOLDER")
        print(f"{'=' * 70}")
        print(f"{'Folder':<50} {'Direct':>8} {'Total':>8}")
        print(f"{'-' * 70}")

        for r in results_sorted:
            indent = "  " * r["depth"]
            name = f"{indent}{r['folder_name']}"
            if len(name) > 48:
                name = name[:45] + "..."
            print(f"{name:<50} {r['direct_images']:>8} {r['total_images']:>8}")

        print(f"{'=' * 70}")
        total = results_sorted[-1]["total_images"] if results_sorted else 0
        print(f"\nTotal images across all folders: {total}\n")

    @staticmethod
    def export_to_csv(results: list[dict], filename: str = "image_counts.csv") -> None:
        results_sorted = sorted(results, key=lambda x: x["path"])

        with open(filename, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["path", "folder_name", "direct_images", "total_images", "depth"]
            )
            writer.writeheader()
            writer.writerows(results_sorted)


if __name__ == "__main__":
    folder_id = sys.argv[1] if len(sys.argv) > 1 else input("Enter the Google Drive folder ID: ").strip()

    if not folder_id:
        raise ValueError("No folder ID provided.")

    service = GoogleDriveAuth.get_drive_service()
    total, results = FileCounter.count_images_per_folder(service, folder_id, folder_name="root")
    FileCounter.print_folder_image_counts(results)
    FileCounter.export_to_csv(results)
