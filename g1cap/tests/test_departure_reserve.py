import math,unittest
from g1cap.source_departure import SourceDeparture
from test_source_departure import sample
class DepartureReserveTests(unittest.TestCase):
 def measured(self,t,lower):
  s=sample(t,True,True);s['sweep']['lower_m']=lower;return s
 def test_three_cm_safety_pass_does_not_satisfy_eight_cm_admission(self):
  c=SourceDeparture(0.,-math.pi/3,sample(0.),planning_margin_m=.08)
  self.assertLess(c.update(.02,self.measured(.02,.04))['navigation'][0],0)
  self.assertIsNone(c.result()['outcome'])
 def test_reserve_then_one_second_settling(self):
  c=SourceDeparture(0.,-math.pi/3,sample(0.),planning_margin_m=.08)
  for i in range(1,52):r=c.update(i*.02,self.measured(i*.02,.08))
  self.assertEqual(r['outcome'],'completed')
  self.assertEqual(r['planning_margin_m'],.08)
 def test_reserve_loss_resets_dwell(self):
  c=SourceDeparture(0.,-math.pi/3,sample(0.),planning_margin_m=.08)
  for i in range(1,30):c.update(i*.02,self.measured(i*.02,.08))
  c.update(.6,self.measured(.6,.07))
  for i in range(31,55):r=c.update(i*.02,self.measured(i*.02,.08))
  self.assertIsNone(r['outcome']);self.assertEqual(r['navigation'],[0.,0.,0.])
 def test_cannot_reduce_runtime_margin_or_accept_invalid_reserve(self):
  for x in [.02,-1,float('nan'),True]:
   with self.assertRaises(ValueError):SourceDeparture(0.,-math.pi/3,sample(0.),planning_margin_m=x)
 def test_reference_selection_uses_planning_margin(self):
  from g1cap.departure_runtime import screen_departure_sweep
  from types import SimpleNamespace
  stop=SimpleNamespace(plane=object());original=stop.plane
  values=iter([.04,.09]);chosen=[]
  def choose(packet,evaluate):
   for i in range(2):
    result=evaluate();chosen.append(result)
    if result['status']=='clear':return result
   return result
  world=SimpleNamespace(loaded_stop_clearance=stop,screen_measured_source=choose)
  result=screen_departure_sweep(world,{},lambda:dict(status='clear',lower_m=next(values)),required_margin_m=.08)
  self.assertEqual(result['lower_m'],.09);self.assertEqual(len(chosen),2);self.assertIs(stop.plane,original)
 def test_explicit_longer_departure_is_still_bounded(self):
  c=SourceDeparture(0.,-math.pi/3,sample(0.),planning_margin_m=.08,max_duration_s=16.)
  for i in range(1,802):r=c.update(i*.02,self.measured(i*.02,.04))
  self.assertEqual(r['outcome'],'failed');self.assertEqual(r['reason'],'departure_timeout')
 def test_invalid_duration_rejected(self):
  for duration in [0.,-1.,float('inf'),True,31.]:
   with self.assertRaises(ValueError):SourceDeparture(0.,-math.pi/3,sample(0.),max_duration_s=duration)
if __name__=='__main__':unittest.main()
