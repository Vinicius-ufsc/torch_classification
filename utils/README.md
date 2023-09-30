### Create csv labels file for image multiclass classification dataset

- here is an example of dataset folder structure:

```plaintext 
├── dataset
│   ├── train
│   │   ├── class1
│   │   │   ├── image1.png
│   │   │   ├── image2.png
│   │   │   └── ...
│   │   ├── class2
│   │   │   ├── image1.png
│   │   │   ├── image2.png
│   │   │   └── ...
│   │   ...
│   ├── val
│   │   ├── class1
│   │   │   ├── image1.png
│   │   │   ├── image2.png
│   │   │   └── ...
│   │ 
```

| Parameter          | Description                                         |
| ------------------ | ----------------------------------------------------|
| --root             | Root directory of the dataset                       |
| --workers          | Maximum number of worker                            |

- Run the provided script (example)
    ```python
    python create_csv.py --root dataset
    ```
