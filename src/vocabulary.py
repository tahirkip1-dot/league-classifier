class Vocabulary:

    def __init__(self, champ_names: list):
        self.names = [name.lower() for name in champ_names]

    def encode(self, name: str):
        name = name.lower()
        return self.names.index(name)

    def decode(self, id: int):
        return self.names[id]
