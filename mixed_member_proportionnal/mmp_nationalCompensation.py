import numpy as np

class Results(object):

    def __init__(self, parties, n_circ=80, n_comp=45):

        self.n_circ = n_circ
        self.n_comp = n_comp
        self.n_seats = self.n_circ + self.n_comp

        self.parties = parties
        self.circ_seats = {k:0 for k in self.parties}
        self.comp_seats = {k:0 for k in self.parties}
        self.distributed_seats = {k:0 for k in self.parties}

        self.seat_percent = {k:0 for k in self.parties}
        self.vote_percent = {k:0 for k in self.parties}

    @property
    def total_distributed_seats(self):
        return sum(self.distributed_seats.values())

    def enter_results(self, vote_percents, circ_seats_won):
        for party, n_seats_won in circ_seats_won:
            self.circ_seats[party] = n_seats_won

        for party, vote_percent in vote_percents:
            self.vote_percent[party] = vote_percent

        assert sum(self.circ_seats.values()) == self.n_circ
        assert sum(self.vote_percent.values()) == 100.

        self.update_percents()

    def _update_distributed_seats(self):
        for party in self.parties:
            self.distributed_seats[party] = self.circ_seats[party] + self.comp_seats[party]

    def update_percents(self):
        self._update_distributed_seats()
        for party in self.parties:
            self.seat_percent[party] = (self.distributed_seats[party] / self.total_distributed_seats) * 100.

    def distribute_comp_seats(self):
        for i in range(self.n_comp):
            rep_distance = [(party, self.vote_percent[party] - self.seat_percent[party]) for party in self.parties]
            sorted_rep_distance = sorted(rep_distance, key=lambda item: item[1])
            worst_represented_party = sorted_rep_distance[-1][0]

            print(f'{i+1} - {worst_represented_party}')

            self.comp_seats[worst_represented_party] += 1
            self.update_percents()

            if i in [15, 30]:
                print(self.seat_percent)

if __name__ == '__main__':

    results = Results(['castor', 'orignal', 'oiseau', 'ours', 'loup'])
    results.enter_results(vote_percents=[('orignal', 35.), ('ours', 24.), ('oiseau', 17.), ('loup', 13.), ('castor', 11.)],
                          circ_seats_won=[('orignal', 41), ('ours', 31), ('oiseau', 5), ('loup', 0), ('castor', 3)])

    results.distribute_comp_seats()
    print()
