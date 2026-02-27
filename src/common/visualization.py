import os

from PIL import ImageDraw


class Visualization:
    @staticmethod
    def draw_boxes(image, predictions: list, color: tuple = (0, 255, 0), width: int = 2):
        draw = ImageDraw.Draw(image)
        for pred in predictions:
            bbox = (int(pred.bbox.minx), int(pred.bbox.miny), int(pred.bbox.maxx), int(pred.bbox.maxy))
            draw.rectangle(bbox, outline=color, width=width)
        return image

    @staticmethod
    def save_visuals(result, output_dir: str, filename: str, color: tuple = (0, 255, 0)) -> str:
        os.makedirs(output_dir, exist_ok=True)
        image = result.image.copy()
        image = Visualization.draw_boxes(image, result.object_prediction_list, color)
        output_path = os.path.join(output_dir, f"{filename}_visuals.jpg")
        image.save(output_path)
        return output_path

    @staticmethod
    def export_sahi_visuals(result, output_dir: str, filename: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        result.export_visuals(
            export_dir=output_dir,
            text_size=1,
            rect_th=2,
            hide_labels=True,
            hide_conf=True,
            file_name=filename,
        )
