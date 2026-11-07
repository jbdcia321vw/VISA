# coding: utf-8
# @email: enoche.chow@gmail.com
"""
Concept Registry: versioned, namespaced concept vocabulary for the
two-stage concept bottleneck recommender.

Concepts are stored in a JSON file of the form:

{
  "version": "beauty_concepts_v1",
  "concepts": [
    {
      "id": 0,
      "name": "category:skincare",
      "namespace": "category",
      "description": "...",
      "display_name": "护肤",
      "synonyms": [...],
      "type": "binary",
      "status": "active"
    }
  ]
}
"""

import json
import os
import re

DEFAULT_REGISTRY_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', 'data', 'beauty', 'concepts', 'concept_registry.json'))

VALID_NAMESPACES = ('category', 'attribute', 'price', 'brand', 'use_case', 'skin_type')
VALID_TYPES = ('binary', 'multiclass')
VALID_STATUS = ('active', 'deprecated')


def canonical_name(name):
    r"""Normalize a concept name to lowercase and replace whitespace with '_'."""
    name = str(name).strip().lower()
    name = re.sub(r'\s+', '_', name)
    return name


class ConceptRegistry(object):
    r"""Load, validate and query a concept registry."""

    def __init__(self, concepts, version='concepts_v1'):
        self.version = version
        self.concepts = list(concepts)          # list of dicts, index == stable id
        self._name2id = {c['name']: c['id'] for c in self.concepts}
        self._id2name = {c['id']: c['name'] for c in self.concepts}
        self.namespaces = sorted({c['namespace'] for c in self.concepts})

    # ---------------- construction ----------------
    @classmethod
    def from_dict(cls, d):
        registry = cls(d['concepts'], version=d.get('version', 'concepts_v1'))
        registry.validate()
        return registry

    @classmethod
    def load(cls, path=None):
        path = path or DEFAULT_REGISTRY_PATH
        if not os.path.isfile(path):
            raise FileNotFoundError('Concept registry not found: {}'.format(path))
        with open(path, 'r', encoding='utf-8') as f:
            d = json.load(f)
        return cls.from_dict(d)

    # ---------------- persistence ----------------
    def save(self, path=None):
        path = path or DEFAULT_REGISTRY_PATH
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        return path

    def to_dict(self):
        return {'version': self.version, 'concepts': self.concepts}

    # ---------------- validation ----------------
    def validate(self):
        ids = [c['id'] for c in self.concepts]
        if len(ids) != len(set(ids)):
            raise ValueError('Concept ids must be unique')
        if sorted(ids) != list(range(len(ids))):
            raise ValueError('Concept ids must be a contiguous range 0..K-1 (stable ids)')
        for c in self.concepts:
            if not re.fullmatch(r'[a-z][a-z0-9_]*:[a-z][a-z0-9_]*', c.get('name', '')):
                raise ValueError('Invalid concept name: {!r}'.format(c.get('name')))
            if ':' not in c['name']:
                raise ValueError('Concept name must be namespaced: {!r}'.format(c['name']))
            ns = c['name'].split(':', 1)[0]
            if c.get('namespace') != ns:
                raise ValueError('namespace field mismatch for {!r}'.format(c['name']))
            if c.get('namespace') not in VALID_NAMESPACES:
                raise ValueError('Invalid namespace: {!r}'.format(c.get('namespace')))
            if c.get('type', 'binary') not in VALID_TYPES:
                raise ValueError('Invalid type for {!r}'.format(c['name']))
            if c.get('status', 'active') not in VALID_STATUS:
                raise ValueError('Invalid status for {!r}'.format(c['name']))

    # ---------------- query ----------------
    def __len__(self):
        return len(self.concepts)

    @property
    def num_concepts(self):
        return len(self.concepts)

    def get_id(self, name):
        r"""Return stable id of concept by canonical name, or None."""
        return self._name2id.get(canonical_name(name))

    def get_name(self, cid):
        return self._id2name.get(cid)

    def names(self):
        return [c['name'] for c in self.concepts]

    def active_names(self):
        return [c['name'] for c in self.concepts if c.get('status', 'active') == 'active']

    def filter_names(self, names):
        r"""Keep only concept names that exist in this registry (drop illegal ones)."""
        valid = set(self._name2id)
        return [canonical_name(n) for n in names if canonical_name(n) in valid]

    def __contains__(self, name):
        return canonical_name(name) in self._name2id

    def __repr__(self):
        return '<ConceptRegistry v{} n={}>'.format(self.version, len(self.concepts))
