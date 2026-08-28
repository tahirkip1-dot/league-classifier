class Vocabulary:
    '''takes riot id to names mapping and builds maps to and from names, riot_ids, model integers'''
    def __init__(self, riot_id_to_name: dict[int, str]):

        # riot id to name dictionary
        riot_id_name = riot_id_to_name.copy()
        riot_id_name[-1] = 'no_ban'
        self.riot_id_name = riot_id_name

        # name to model integer dictionary
        names = list(riot_id_name.values()) + ['masked']
        self.name_id = dict(zip(names, range(len(names))))

        mask_id = self.name_id['masked']
        self.mask_id = mask_id

        no_ban_id = self.name_id['no_ban']
        self.no_ban_id = no_ban_id

        # model integer to name dictionary
        self.id_name = {value:key for key, value in self.name_id.items()}


    def riot_id_to_id(self, riot_id: int):
        name = self.riot_id_name[riot_id]
        return self.name_id[name]

    def name_to_id(self, name: str):
        return self.name_id[name]


    def id_to_name(self, id: int):
        return self.id_name[id]

    def __len__(self):
        return len(self.name_id.items())
    
    
