from loader import ECoGDataset

class TimeOfDayDataset(ECoGDataset):
    def __init__(self, args, root, path, bands, fs, new_fs, start_time, sample_secs = 2):
        super().__init__(args, root, path, bands, fs, new_fs, sample_secs)
        self.start_time = start_time
        
    def sample_data(self):
        sample_time = self.start_time + self.sample_secs * self.index
        signal = super().sample_data()
        return (signal, sample_time)