class Vocabulary:
    '''adds the masked champion, dont add manually beforehand'''
    def __init__(self, champ_names: list[str]):
        self.names = [name.lower() for name in champ_names] + ['masked']

    def encode(self, name: str):
        name = name.lower()
        return self.names.index(name)

    def decode(self, id: int):
        return self.names[id]

    def mask_id(self):
        return self.encode('masked')

    def __len__(self):
        return len(self.names)
