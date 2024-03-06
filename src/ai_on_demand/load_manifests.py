from aiod_registry import load_manifests
from ai_on_demand.utils import load_settings


def activate_plugin(ctx):
    print("Plugin activated!")
    all_manifests = load_manifests(filter_access=True)
    plugin_settings = load_settings()

    def get_manifests():
        return all_manifests

    ctx.register_command("ai-on-demand.get_manifests", get_manifests)

    def get_settings():
        return plugin_settings

    ctx.register_command("ai-on-demand.get_settings", get_settings)
