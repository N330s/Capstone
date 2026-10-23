import unittest
from learning.vla.inference_cadence import anchor_noise_seed,audit_cadence_chunks,resource_violations,continuation_state


class CadenceTests(unittest.TestCase):
    def test_common_anchor_same_seed_regardless_of_prefix(self):
        for prefix in (1,5,25):
            self.assertEqual(anchor_noise_seed(1000,25),1025)
            chunks=[{'start_step':i,'executed_steps':min(prefix,53-i),'noise_seed':1000+i}
                    for i in range(0,53,prefix)]
            self.assertEqual(audit_cadence_chunks(chunks,prefix=prefix,steps=53,seed=1000,schedule='step-keyed')['commands'],53)

    def test_wrong_seed_or_shifted_prefix_rejected(self):
        valid=[{'start_step':0,'executed_steps':5,'noise_seed':1000},
               {'start_step':5,'executed_steps':2,'noise_seed':1005}]
        for key,value in (('start_step',6),('noise_seed',999),('executed_steps',1)):
            broken=[dict(c) for c in valid]; broken[1][key]=value
            with self.assertRaises(ValueError):
                audit_cadence_chunks(broken,prefix=5,steps=7,seed=1000,schedule='step-keyed')

    def test_seed_domain(self):
        for seed,index in ((True,1),(-1,0),(0,-1),(2**63-1,1)):
            with self.assertRaises(ValueError): anchor_noise_seed(seed,index)

    def test_disk_guard_is_distinct_from_gpu_and_ram(self):
        args={'temperature_c':65,'gpu_free_mib':4500,'ram_available_bytes':12*2**30,'disk_free_bytes':23*2**30}
        self.assertEqual(resource_violations(**args),['Disk free below 25 GiB reserve'])
        self.assertEqual(resource_violations(**{**args,'disk_free_bytes':25*2**30}),[])

    def test_thermal_guard_independent_of_disk(self):
        self.assertEqual(resource_violations(temperature_c=85,gpu_free_mib=4500,
            ram_available_bytes=12*2**30,disk_free_bytes=28*2**30),['GPU temperature at/above 85 C'])

    def resume_fixture(self):
        cases=[{'episode':str(i),'prefix':25,'mode':'replay','noise_schedule':'sequential'} for i in range(3)]
        plan={'protocol':'fixed-actor-cadence-v1','total_limit_s':3600,'cases':cases}
        rows=[{**cases[0],'pickup_success':True}]
        fail={'completed':False,'error':"RuntimeError('GPU thermal/memory or RAM/disk safety guard')",
              'elapsed_s':217.812,'conditions':rows,'remaining_cases_not_run':cases[1:]}
        return plan,fail,rows

    def test_continuation_charges_prior_clock(self):
        p,f,r=self.resume_fixture(); state=continuation_state(p,f,r)
        self.assertEqual(state['charged_prior_s'],218)
        self.assertEqual(state['restart_interrupted_condition'],p['cases'][1])

    def test_no_silent_restart_after_algorithm_failure(self):
        p,f,r=self.resume_fixture(); f['error']='wrong weights'
        with self.assertRaises(ValueError): continuation_state(p,f,r)

    def test_no_reordering_or_dropping_prior_conditions(self):
        p,f,r=self.resume_fixture(); f['remaining_cases_not_run']=list(reversed(f['remaining_cases_not_run']))
        with self.assertRaises(ValueError): continuation_state(p,f,r)


if __name__=='__main__': unittest.main()
