#include "wav_io.h"
#include <cassert>
#include <filesystem>
#include <unistd.h>
static void put(std::ofstream& f, uint32_t x, int n) { for(int i=0;i<n;++i) f.put(char(x >> (8*i))); }
static void fixture(const std::string& path, uint16_t format, uint16_t bits, const std::vector<uint8_t>& bytes, uint16_t channels=1) {
    std::ofstream f(path, std::ios::binary);
    f.write("RIFF",4); put(f,36+bytes.size()+(bytes.size()&1),4); f.write("WAVEfmt ",8); put(f,16,4);
    put(f,format,2); put(f,channels,2); put(f,48000,4); put(f,48000*channels*bits/8,4); put(f,channels*bits/8,2); put(f,bits,2);
    f.write("data",4); put(f,bytes.size(),4); f.write(reinterpret_cast<const char*>(bytes.data()),bytes.size()); if(bytes.size()&1) f.put(0);
}
int main() {
    char dir[]="/tmp/earshot-wav-test-XXXXXX"; assert(mkdtemp(dir));
    std::string path=std::string(dir)+"/sample.wav"; std::vector<float> out; int rate=0;
    // Three PCM24 samples occupy nine bytes. The old reader wrote nine bytes
    // into four int16 values (eight bytes), and decoded even samples wrongly.
    fixture(path,1,24,{0,0,0,0xff,0xff,0x7f,0,0,0x80});
    assert(WavWriter::readWavData(path,out,rate)); assert(rate==48000); assert(out.size()==3);
    assert(out[0]==0); assert(out[1]>0.9999f); assert(out[2]==-1);
    fixture(path,1,16,{0,0x80,0xff,0x7f}); assert(WavWriter::readWavData(path,out,rate)); assert(out.size()==2 && out[0]==-1 && out[1]>0.999f);
    fixture(path,1,32,{0,0,0,0x80}); assert(WavWriter::readWavData(path,out,rate)); assert(out[0]==-1);
    fixture(path,3,32,{0,0,0,0x3f}); assert(WavWriter::readWavData(path,out,rate)); assert(out[0]==0.5f);
    fixture(path,1,16,{0}); assert(!WavWriter::readWavData(path,out,rate));
    fixture(path,3,32,{0,0,0}); assert(!WavWriter::readWavData(path,out,rate));
    fixture(path,1,24,{0,0,0,0,0,0x80},2); assert(WavWriter::readWavData(path,out,rate)); assert(out.size()==1 && out[0]==-0.5f);
    fixture(path,1,24,{0,0,0}); std::filesystem::resize_file(path,20); assert(!WavWriter::readWavData(path,out,rate));
    std::filesystem::remove_all(dir);
}
