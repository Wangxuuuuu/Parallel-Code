#include "PCFG.h"
#include <chrono>
#include <fstream>
#include "md5.h"
#include <iomanip>
#include <vector>
using namespace std;
using namespace chrono;

// 编译指令如下：
// g++ correctness.cpp train.cpp guessing.cpp md5.cpp -o test.exe
// g++ -march=armv8-a correctness.cpp train.cpp guessing.cpp md5.cpp -o test.exe


// 通过这个函数，你可以验证你实现的SIMD哈希函数的正确性
int main()
{   string input[4] = {
    "abcdefg",
    "abcdefg",
    "abcdefg",
    "abcdefg"
    };
    // 用于存储每个消息的哈希值
    bit32 state[4][4]{};

    MD5Hash(input, state);

    for (int i1 = 0; i1 < 4; i1 += 1){
        for(int j=0;j<4;j++){
        cout << std::setw(8) << std::setfill('0') << hex << state[i1][j];
        }
        cout<<endl;
    }

    cout << endl;
}
