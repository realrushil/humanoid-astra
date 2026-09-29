"""Explicit experiment language must reach policy and evidence identically."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

class AcquisitionInstructionTests(unittest.TestCase):
    def owner(self):
        tree=ast.parse(Path('g1cap/arena_world.py').read_text())
        owner=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ArenaWorld')
        method=next((n for n in owner.body if isinstance(n,ast.FunctionDef) and n.name=='configure_acquisition'),None)
        self.assertIsNotNone(method,'configure acquisition must record the exact instruction sent')
        scope={};exec(compile(ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[])),'owner.py','exec'),scope)
        return scope['configure_acquisition']
    def test_default_retains_native_instruction(self):
        configure=self.owner();obj=SimpleNamespace(policy=Mock());native=Mock();native.get_language_instruction.return_value='Native task'
        configure(obj,{},native)
        obj.policy.set_task_description.assert_called_once_with('Native task')
        self.assertEqual(obj.policy_instruction,'Native task');obj.policy.reset.assert_called_once_with()
    def test_override_reaches_policy_without_reading_native_task(self):
        configure=self.owner();obj=SimpleNamespace(policy=Mock());native=Mock()
        configure(obj,{'acquisition_instruction':'Pick up the brown box.'},native)
        obj.policy.set_task_description.assert_called_once_with('Pick up the brown box.')
        native.get_language_instruction.assert_not_called();self.assertEqual(obj.policy_instruction,'Pick up the brown box.')
    def test_invalid_instruction_rejects_before_policy_mutation(self):
        configure=self.owner()
        for value in (None,False,12,'','   ','x'*513):
            obj=SimpleNamespace(policy=Mock())
            with self.assertRaises(ValueError):configure(obj,{'acquisition_instruction':value},Mock())
            self.assertFalse(obj.policy.mock_calls)
