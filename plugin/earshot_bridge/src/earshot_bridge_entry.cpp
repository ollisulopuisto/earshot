#include <clap/clap.h>

bool earshot_bridge_entry_init(const char *plugin_path);
void earshot_bridge_entry_deinit(void);
const void *earshot_bridge_entry_get_factory(const char *factory_id);

extern "C" {
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wattributes"
#endif

CLAP_EXPORT const struct clap_plugin_entry clap_entry = {
    CLAP_VERSION_INIT,
    earshot_bridge_entry_init,
    earshot_bridge_entry_deinit,
    earshot_bridge_entry_get_factory
};

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
}
