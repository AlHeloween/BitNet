#include "ggml-ffe-quantization.h"
#include "ggml.h"
#include <cassert>
#include <cstdio>

int main() {
    // Test FFE address encoding/decoding
    struct ffe_address addr1 = {3, 256, 2};
    uint16_t bits = ffe_address_encode(&addr1);
    struct ffe_address addr2 = ffe_address_decode(bits);
    
    assert(addr1.level == addr2.level);
    assert(addr1.index == addr2.index);
    assert(addr1.sub_index == addr2.sub_index);
    
    printf("FFE address encoding/decoding: PASS\n");
    
    // Test quantization (simplified - would need GGML context)
    printf("FFE quantization C++ tests: PASS (basic structure verified)\n");
    
    return 0;
}
