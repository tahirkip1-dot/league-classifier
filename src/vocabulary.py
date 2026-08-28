class Vocabulary:
    '''takes riot id to names mapping and builds maps to and from names, riot_ids, model integers'''

    @staticmethod
    def _normalise_name(name: str):
        return ''.join(character for character in name.casefold() if character.isalnum())

    def __init__(self, riot_id_to_name: dict[int, str]):

        # riot id to name dictionary
        riot_id_name = riot_id_to_name.copy()
        riot_id_name[-1] = 'no_ban'
        self.riot_id_name = riot_id_name

        # normalised name to model integer dictionary
        names = list(riot_id_name.values()) + ['masked']
        normalised_names = [self._normalise_name(name) for name in names]
        self.name_id = dict(zip(normalised_names, range(len(normalised_names))))

        mask_id = self.name_id[self._normalise_name('masked')]
        self.mask_id = mask_id

        no_ban_id = self.name_id[self._normalise_name('no_ban')]
        self.no_ban_id = no_ban_id

        # model integer to name dictionary
        self.id_name = dict(zip(range(len(names)), names))


    def riot_id_to_id(self, riot_id: int):
        name = self.riot_id_name[riot_id]
        normalised_name = self._normalise_name(name)
        return self.name_id[normalised_name]

    def name_to_id(self, name: str):
        normalised_name = self._normalise_name(name)
        return self.name_id[normalised_name]

    def id_to_name(self, id: int):
        return self.id_name[id]

    def __len__(self):
        return len(self.name_id.items())
    
    
