import importlib

def import_yaml_lib(cfg, target_attr):
        dataset_name = cfg.get("dataset_name", None)
        module_name, class_name = cfg.get(target_attr).rsplit(".", 1)
        if module_name.startswith("task_utils"):
                module = importlib.import_module(f"tasks.{dataset_name}.{module_name}")
        else:
                module = importlib.import_module(module_name)
        return getattr(module, class_name)