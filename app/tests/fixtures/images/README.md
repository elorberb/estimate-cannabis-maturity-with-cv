# Sample Images for Testing

Real cannabis trichome photos from experiment_1, one per measurement day.
Use these to test the `/api/v1/analyze` endpoint without needing your own images.

| File | Date | Notes |
|---|---|---|
| `sample_day1.jpg` | 2024-05-30 | Early flowering |
| `sample_day3.jpg` | 2024-06-06 | Early-mid |
| `sample_day4.jpg` | 2024-06-10 | Mid |
| `sample_day5.jpg` | 2024-06-13 | Mid |
| `sample_day6.jpg` | 2024-06-17 | Mid-late |
| `sample_day7.jpg` | 2024-06-20 | Late |
| `sample_day9.jpg` | 2024-06-27 | Final measurement |

## Usage

```bash
# Hit the analyze endpoint with a sample image
curl -X POST "http://localhost:8000/api/v1/analyze?device_id=test" \
  -F "file=@sample_day1.jpg"
```
