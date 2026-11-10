#include "../src/earshot_bridge_plugin.cpp"
#include <cassert>
#include <sys/stat.h>
static void param(earshot_bridge_plug_t* p, int id, double value) {
    clap_event_param_value_t ev{}; ev.header.type=CLAP_EVENT_PARAM_VALUE; ev.param_id=id; ev.value=value;
    bridge_handle_param_event(p,&ev.header);
}
static void wait_job(earshot_bridge_plug_t* p) {
    for(int i=0;i<1000 && p->job_pending.load();++i) std::this_thread::sleep_for(std::chrono::milliseconds(2));
    assert(!p->job_pending.load());
}
int main() {
    char tools[]="/tmp/earshot-test-tools-XXXXXX"; assert(mkdtemp(tools));
    const std::string script=std::string(tools)+"/earshot";
    { std::ofstream f(script); f << "#!/bin/sh\nsleep 0.05\ncp \"$2\" \"$4\"\n"; }
    chmod(script.c_str(),0700); const std::string path=std::string(tools)+":"+getenv("PATH"); setenv("PATH",path.c_str(),1);
    clap_host_t host{}; auto *plugin=create_earshot_bridge_plugin(&host); assert(plugin);
    auto *p=static_cast<earshot_bridge_plug_t*>(plugin->plugin_data);
    struct stat st{}; assert(stat(p->bridge_dir.c_str(),&st)==0 && (st.st_mode&0777)==0700);
    const std::string dir=p->bridge_dir;
    bridge_activate(plugin,48000,1,4);
    float input[4]={.25f,.25f,.25f,.25f}, output[4]{}; float *inputs[]={input}, *outputs[]={output};
    clap_audio_buffer_t in{},out{}; in.channel_count=out.channel_count=1; in.data32=inputs; out.data32=outputs;
    clap_event_transport_t transport{}; transport.flags=CLAP_TRANSPORT_HAS_SECONDS_TIMELINE|CLAP_TRANSPORT_IS_PLAYING;
    clap_process_t process{}; process.frames_count=4; process.audio_inputs=&in; process.audio_outputs=&out;
    process.audio_inputs_count=process.audio_outputs_count=1; process.transport=&transport;
    param(p,PARAM_CAPTURE,1); bridge_process(plugin,&process); assert(p->captured_buffer.size()==4);
    param(p,PARAM_RESTORE,1); bridge_process(plugin,&process); assert(!p->capturing.load());
    assert(p->job_pending.load());
    // A capture arriving while the immutable job is in use must not clear it.
    param(p,PARAM_CAPTURE,1); assert(!p->capturing.load() && p->captured_buffer.size()==4);
    // Process while the worker owns the capture, stressing publication while
    // audio still runs. No call may resize or clear the worker's capture.
    for(int i=0;i<1000 && p->job_pending.load();++i) {
        bridge_process(plugin,&process);
        assert(p->captured_buffer.size()==4);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    wait_job(p);
    std::vector<float> captured; int recorded_rate=0;
    assert(WavWriter::readWavData(dir+"/selection_in.wav",captured,recorded_rate));
    assert(recorded_rate==48000 && captured.size()==4 && captured[0]==.25f);
    for(auto &x: input) x=0;
    transport.song_pos_seconds=CLAP_SECTIME_FACTOR; // completion at 1 s: do not play sample zero here
    bridge_process(plugin,&process); for(auto x:output) assert(x==0);
    bridge_reset(plugin); transport.song_pos_seconds=0; bridge_process(plugin,&process);
    for(auto x:output) assert(x==.25f); // seek back to the captured position realigns the result
    transport.flags=CLAP_TRANSPORT_HAS_SECONDS_TIMELINE; bridge_process(plugin,&process);
    for(auto x:output) assert(x==0); // stopped means passthrough
    transport.flags|=CLAP_TRANSPORT_IS_PLAYING;
    for(auto &x:input) x=.75f;
    param(p,PARAM_CAPTURE,1); bridge_process(plugin,&process); // capture while the previous result exists
    assert(p->captured_buffer[0]==.75f);
    param(p,PARAM_RESTORE,1); bridge_process(plugin,&process); wait_job(p);
    for(auto &x:input) x=0;
    bridge_process(plugin,&process); for(auto x:output) assert(x==.75f);
    process.transport=nullptr; bridge_process(plugin,&process); for(auto x:output) assert(x==0);
    // A slow CLI must be cancellable when the host removes the plugin.
    { std::ofstream f(script); f << "#!/bin/sh\nsleep 30\n"; }
    param(p,PARAM_RESTORE,1); bridge_process(plugin,&process);
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    auto before = std::chrono::steady_clock::now();
    bridge_deactivate(plugin); bridge_destroy(plugin);
    assert(std::chrono::steady_clock::now() - before < std::chrono::seconds(2));
    assert(!std::filesystem::exists(dir)); std::filesystem::remove_all(tools);
}
