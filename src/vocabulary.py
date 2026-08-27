class Vocabulary:
    '''takes riot id to names mapping and builds maps to and from names, riot_ids, model integers'''
    def __init__(self, riot_id_to_name: dict[int, str]):

        # riot id to name dictionary
        riot_id_name = riot_id_to_name
        riot_id_name[-1] = 'no_ban'
        self.riot_id_name = riot_id_name

        # name to model integer dictionary
        names = list(riot_id_name.values()) + ['masked']
        self.name_id = dict(zip(names, range(len(names))))


    def riot_id_to_id(self, riot_id: int):
        name = self.riot_id_name[riot_id]
        return self.name_id[name]

    def __len__(self):
        return len(self.name_id.items())
    
'''
    def decode(self, id: int):
        return self.names[id]

    def mask_id(self):
        return self.encode('masked')
'''
    
