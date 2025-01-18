import copy
import random
from typing import Any, Callable, Optional
import numpy as np
from pydantic import BaseModel, ConfigDict
from copy import deepcopy
from sparse._sparse_array import SparseArray

class State(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    am: SparseArray
    index: int


class Entity(BaseModel):
    name: str
    type: str
    idx: Optional[int] = None

    def __hash__(self):
        return hash((self.name, self.type))


class Coordenate(BaseModel):
    name: str
    type: str
    idx: Optional[int] = None

    def __hash__(self):
        return hash((self.name, self.type))


class UnitState(BaseModel):
    entity: Entity
    coordenate: Coordenate


class EntityInCoordenateState(State):
    am: SparseArray

    @property
    def attr_as_set(self):
        attr = []
        for unit in self.am:
            attr.append((unit.entity, unit.coordenate))
        return set(attr)

    def create_transition(
        self,
        num_transitions: int,
        condition: Callable,
    ) -> SparseArray:
        """
        Creates a delta of actor-location pairs based on specified conditions.
        Args:
            num_transitions (int): The number of transitions (actor-location pairs) to 
            create.
            coordenate (list[str]): A list of possible coordenate.
            condition (Callable): A callable that takes a pair (actor, location) 
            and returns a boolean.
        Returns:
           SparseArray: A SparseArray objects representing the new state.
        """
        
        while True:
            next_am = deepcopy(self.am)
            e = random.choices(list(next_am.data.keys()), k=num_transitions)
            for i_e in e:
                x, y = i_e[0], i_e[1]
                #get a different coordenate different from current
                next_y = random.choice(list(set([x for x in range(next_am.shape[1])]) - set([y])))
                next_am[x, next_y] = 1
                next_am[i_e] = 0

            # check if the delta satisfies the conditions
            if condition(next_am):
                pass
            else:
                continue

            f = self.validate_next(next_am)

            return next_am, f

    def create_state_from_delta(self, j: int, delta: list[UnitState]):
        new_attr = copy.deepcopy(self.am)
        new_state = EntityInCoordenateState(attr=new_attr, index=j)
        for delta_i in delta:
            for unit in new_state.attr:
                if unit.entity == delta_i.entity:
                    unit.coordenate = delta_i.coordenate
        return new_state

    def get_entity_coordenate(self, entity: Entity):
        for unit in self.am:
            if unit.entity == entity:
                return unit.coordenate
        return None

    def get_entities_in_coodenate(self, coordenate: Coordenate):
        entities = []
        for unit in self.am:
            if unit.coordenate == coordenate:
                entities.append(unit.entity)
        return entities

    def validate_next(self, next_am):
        flag = True
        # Check that the entitis is no in more than 1
        # coodenate
        es = np.arange(next_am.shape[0])
        for e in es:
            if sum(next_am[e,:]) != 1:
                flag = False
        return flag    