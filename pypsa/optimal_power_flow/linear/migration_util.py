from collections import Counter
from io import StringIO
from itertools import chain
from types import SimpleNamespace as Namespace


class ProxyStats:
    def __init__(self, attributes):
        self.attributes = set(attributes)
        self.get_counts = Counter()
        self.set_counts = Counter()
        self.del_counts = Counter()

    def __str__(self):
        return f"ProxyStats(attrs={self.attributes}, new_attrs={self.new_attrs})"

    def report(self):
        print("ProxyStats")
        print(f"attrs={self.attributes}, new_attrs={self.new_attrs}")
        print("GET")
        for key, count in self.get_counts.most_common():
            print(f"\t-{key} ({count})")
        print("SET")
        for key, count in self.set_counts.most_common():
            print(f"\t-{key} ({count})")
        print("DEL")
        for key, count in self.del_counts.most_common():
            print(f"\t-{key} ({count})")

    @property
    def new_attrs(self):
        return set(self.set_counts) - self.attributes

    def add_get(self, key):
        self.get_counts[key] += 1

    def add_set(self, key):
        self.set_counts[key] += 1

    def add_del(self, key):
        self.del_counts[key] += 1


class Proxy:
    def __init__(self, subject, **kwargs):
        kwargs["_subject"] = subject
        kwargs["_stats"] = ProxyStats(subject.__dict__)
        self.__dict__.update(kwargs)

    def _repr(self, show_private=False):
        def show(key):
            return show_private or not key.startswith('_')

        def attr_to_str(value):
            if isinstance(value, property):
                try:
                    return value.fget(self)
                except AttributeError:
                    pass
            return str(value)

        cls_name = type(self._subject).__name__
        obj_attr = self.__dict__.items()
        cls_attr = ((k, v) for k, v in type(self).__dict__.items() if isinstance(v, property))
        attr_str = ', '.join(f"{k}={attr_to_str(v)}" for k, v in chain(obj_attr, cls_attr) if show(k))

        return f"{cls_name}Proxy({attr_str})"

    def __str__(self):
        return self._repr(show_private=False)

    def __repr__(self):
        return self._repr(show_private=True)

    def __getattr__(self, key):
        self._stats.add_get(key)
        return getattr(self._subject, key)

    def __setattr__(self, key, value):
        if key in self.__dict__ or key in self.__class__.__dict__:
            super().__setattr__(key, value)
        else:
            self._stats.add_set(key)
            setattr(self._subject, key, value)

    def __delattr__(self, key):
        if key in self.__dict__ or key in self.__class__.__dict__:
            super().__delattr__(key)
        else:
            self._stats.add_del(key)
            delattr(self._subject, key)


class TracingProxy(Proxy):
    def __init__(self, subject, file=None, log_get=True, log_set=True, log_del=True):
        super().__init__(
            subject=subject,
            _log=file or StringIO(),
            _log_get=log_get,
            _log_set=log_set,
            _log_del=log_del,
        )

    def __getattr__(self, key):
        if self._log_get:
            print(f"GETATTR\t{key}", file=self._log)
        return super().__getattr__(key)

    def __setattr__(self, key, value):
        if self._log_set:
            print(f"SETATTR\t{key}\t{repr(value)}", file=self._log)
        return super().__setattr__(key, value)

    def __delattr__(self, key):
        if self._log_del:
            print(f"DELATTR\t{key}", file=self._log)
        super().__delattr__(key)

    def bookmark(self, tag):
        tag = f" {tag} "
        print(f"BOOKMARK  {tag:-^70}", file=self._log)

    @property
    def log(self) -> StringIO:
        return self._log


def proxy_property(alias: str, store: str):
    return property(
        fget=lambda self: getattr(getattr(self, store), alias),
        fset=lambda self, value: setattr(getattr(self, store), alias, value),
        fdel=lambda self: delattr(getattr(self, store), alias)
    )


class NetworkProxy(Proxy):
    def __init__(self, network, lp=None, solution=None, **kwargs):
        super().__init__(
            subject=network,
            _lp=lp or Namespace(),
            _solution=solution or Namespace(),
            _params=Namespace(**kwargs)
        )


class NetworkProxyPypsa(NetworkProxy):
    """Mocks the legacy interface of the `Network` class but decouples data internally"""
    # params
    _multi_invest = proxy_property("multi_invest", "_params")
    # lp
    _cCounter = proxy_property("_c_counter", "_lp")
    _xCounter = proxy_property("_x_counter", "_lp")
    vars = proxy_property("vars", "_lp")
    cons = proxy_property("cons", "_lp")
    variables = proxy_property("variables", "_lp")
    constraints = proxy_property("constraints", "_lp")
    basis_fn = proxy_property("_basis_fn", "_lp")
    objective_f = proxy_property("_objective_f", "_lp")
    constraints_f = proxy_property("_constraints_f", "_lp")
    bounds_f = proxy_property("_bounds_f", "_lp")
    binaries_f = proxy_property("_binaries_f", "_lp")
    # results
    sols = proxy_property("sols", "_lp")
    solutions = proxy_property("solutions", "_lp")
    duals = proxy_property("duals", "_lp")
    dualvalues = proxy_property("dual_values", "_lp")
    objective = proxy_property("objective", "_solution")
