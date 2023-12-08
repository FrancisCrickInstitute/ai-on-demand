from aiod_registry import load_manifests


def activate_plugin(ctx):
    print("Plugin activated!")
    all_manifests = load_manifests(filter_access=True)

    def get_manifests():
        return all_manifests

    ctx.register_command("ai-on-demand.get_manifests", get_manifests)
