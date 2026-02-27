from src.datasets_and_annotations.segmentsai_handler import SegmentsAIHandler

if __name__ == "__main__":
    segmentsai_handler = SegmentsAIHandler()
    dest_dataset_identifier = "etaylor/cannabis_patches_all_11_4_24"

    segmentsai_handler.copy_dataset_contents(
        source_dataset_id="etaylor/cannabis_patches_train",
        destination_dataset_id=dest_dataset_identifier,
        verbose=True,
        only_patches=True,
    )
    segmentsai_handler.copy_dataset_contents(
        source_dataset_id="etaylor/cannabis_patches_test",
        destination_dataset_id=dest_dataset_identifier,
        verbose=True,
        only_patches=True,
    )
