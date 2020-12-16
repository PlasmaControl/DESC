import unittest

from desc.plotting import Plot

class TestPlot(unittest.TestCase):

    def setUp(self):
        self.names = ['B', '|B|', 'B^zeta', 'B_zeta', 'B_r', 'B^zeta_r',
                'B_zeta_r', 'B**2', 'B_r**2', 'B^zeta**2', 'B_zeta**2',
                'B^zeta_r**2', 'B_zeta_r**2']
        self.bases = ['B', '|B|', 'B', 'B', 'B', 'B',
                'B', 'B', 'B', 'B', 'B',
                'B', 'B']
        self.sups = ['', '', 'zeta', '', '', 'zeta',
                '', '', '', 'zeta', '',
                'zeta', '']
        self.subs = ['', '', '', 'zeta', '', '',
                'zeta', '', '', '', 'zeta',
                '', 'zeta']
        self.ds = ['', '', '', '', 'r', 'r',
                'r', '', 'r', '', '',
                'r', 'r']
        self.pows = ['', '', '', '', '', '',
                '', '2', '2', '2', '2',
                '2', '2']
        self.name_dicts = []
        self.plot = Plot()
        for name in self.names:
            self.name_dicts.append(self.plot.format_name(name))

    def test_name_dict(self):
        self.assertTrue(all([self.name_dicts[i]['base'] == self.bases[i] for i in
            range(len(self.names))]))
        self.assertTrue(all([self.name_dicts[i]['sups'] == self.sups[i] for i in
            range(len(self.names))]))
        self.assertTrue(all([self.name_dicts[i]['subs'] == self.subs[i] for i in
            range(len(self.names))]))
        self.assertTrue(all([self.name_dicts[i]['d'] == self.ds[i] for i in
            range(len(self.names))]))
        self.assertTrue(all([self.name_dicts[i]['power'] == self.pows[i] for i in
            range(len(self.names))]))

    def test_name_label(self):
        labels = [self.plot.name_label(nd) for nd in self.name_dicts]
        print(labels)
        self.assertTrue(all([label[0] == '$' and label[-1] == '$' for label in labels]))
        self.assertTrue(all(['/dr' in labels[i] for i in range(len(labels)) if
            self.name_dicts[i]['d'] != '']))
        self.assertTrue(all(['^{' not in labels[i] for i in range(len(labels))
            if self.name_dicts[i]['sups'] == '' and self.name_dicts[i]['power'] == '']))
        self.assertTrue(all(['_{' not in labels[i] for i in range(len(labels))
            if self.name_dicts[i]['subs'] == '']))
