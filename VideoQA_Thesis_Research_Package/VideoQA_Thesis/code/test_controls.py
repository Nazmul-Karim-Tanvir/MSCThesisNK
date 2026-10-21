import unittest
from collections import Counter
from data import norm,make_subsets,stratum
from bootstrap import paired_cluster

class Controls(unittest.TestCase):
    def test_normalization(self):
        self.assertEqual(norm('  A  GLASS  '),'a glass')
    def test_matched_strata_and_video(self):
        rows=[dict(uid=str(i),video_id=str(i%3),question='what object?',answer='glass' if i%2 else 'cup',
                   question_type='descriptive',reliability=i/30) for i in range(30)]
        counts={'glass':10,'cup':100}
        selected,control=make_subsets(rows,counts,.5,42,True)
        key=lambda r:(stratum(r,counts),r['video_id'])
        self.assertEqual(Counter(map(key,selected)),Counter(map(key,control)))
        self.assertEqual(len({r['uid'] for r in selected}),len(selected))
        again,_=make_subsets(rows,counts,.5,42,True)
        self.assertEqual(selected,again)
    def test_bootstrap_identical_models(self):
        rows=[dict(uid=str(i),video_id=str(i//2),correct=i%2,target=0) for i in range(12)]
        result=paired_cluster(rows,list(reversed(rows)),100,42)
        self.assertEqual(result['accuracy_difference'],0)
        self.assertEqual(result['ci95'],[0,0])
    def test_pair_mismatch_rejected(self):
        with self.assertRaises(ValueError):paired_cluster([dict(uid='a')],[dict(uid='b')],100)

if __name__=='__main__':unittest.main()
