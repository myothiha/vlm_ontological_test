class Dataset():
    def __init__(self, data: list):
        self.set_data(data)

    @staticmethod
    def assert_valid_data(data):
        assert isinstance(data, list), "Data must be a list."
        for i, item in enumerate(data):
            assert isinstance(item, dict), f"Item at index {i} must be a dictionary."
            assert "text" in item, f"Missing 'text' key in item at index {i}."
            assert "image" in item, f"Missing 'image' key in item at index {i}."

    def get_data(self):
        return self.data

    def set_data(self, new_data):
        self.assert_valid_data(new_data)
        self.data = new_data

    def __repr__(self):
        return f"Dataset(data={self.data})"

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def get_all_texts(self):
        """Return a list of all 'text' fields in the dataset."""
        return [item["text"] for item in self.data]

    def get_all_images(self):
        """Return a list of all 'image' fields in the dataset."""
        return [item["image"] for item in self.data]