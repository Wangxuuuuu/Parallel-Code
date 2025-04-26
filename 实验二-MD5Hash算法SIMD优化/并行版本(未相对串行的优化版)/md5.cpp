#include "md5.h"
#include <iomanip>
#include <assert.h>
#include <chrono>
#include <arm_neon.h>


using namespace std;
using namespace chrono;

// T 常量（第一轮，16个常量）
const uint32x4_t T1_1 = vdupq_n_u32(0xd76aa478);
const uint32x4_t T1_2 = vdupq_n_u32(0xe8c7b756);
const uint32x4_t T1_3 = vdupq_n_u32(0x242070db);
const uint32x4_t T1_4 = vdupq_n_u32(0xc1bdceee);
const uint32x4_t T1_5 = vdupq_n_u32(0xf57c0faf);
const uint32x4_t T1_6 = vdupq_n_u32(0x4787c62a);
const uint32x4_t T1_7 = vdupq_n_u32(0xa8304613);
const uint32x4_t T1_8 = vdupq_n_u32(0xfd469501);
const uint32x4_t T1_9 = vdupq_n_u32(0x698098d8);
const uint32x4_t T1_10 = vdupq_n_u32(0x8b44f7af);
const uint32x4_t T1_11 = vdupq_n_u32(0xffff5bb1);
const uint32x4_t T1_12 = vdupq_n_u32(0x895cd7be);
const uint32x4_t T1_13 = vdupq_n_u32(0x6b901122);
const uint32x4_t T1_14 = vdupq_n_u32(0xfd987193);
const uint32x4_t T1_15 = vdupq_n_u32(0xa679438e);
const uint32x4_t T1_16 = vdupq_n_u32(0x49b40821);

// T 常量（第二轮，Round 2）
const uint32x4_t T2_1  = vdupq_n_u32(0xf61e2562);
const uint32x4_t T2_2  = vdupq_n_u32(0xc040b340);
const uint32x4_t T2_3  = vdupq_n_u32(0x265e5a51);
const uint32x4_t T2_4  = vdupq_n_u32(0xe9b6c7aa);
const uint32x4_t T2_5  = vdupq_n_u32(0xd62f105d);
const uint32x4_t T2_6  = vdupq_n_u32(0x02441453);
const uint32x4_t T2_7  = vdupq_n_u32(0xd8a1e681);
const uint32x4_t T2_8  = vdupq_n_u32(0xe7d3fbc8);
const uint32x4_t T2_9  = vdupq_n_u32(0x21e1cde6);
const uint32x4_t T2_10 = vdupq_n_u32(0xc33707d6);
const uint32x4_t T2_11 = vdupq_n_u32(0xf4d50d87);
const uint32x4_t T2_12 = vdupq_n_u32(0x455a14ed);
const uint32x4_t T2_13 = vdupq_n_u32(0xa9e3e905);
const uint32x4_t T2_14 = vdupq_n_u32(0xfcefa3f8);
const uint32x4_t T2_15 = vdupq_n_u32(0x676f02d9);
const uint32x4_t T2_16 = vdupq_n_u32(0x8d2a4c8a);

// T 常量（第三轮，Round 3）
const uint32x4_t T3_1  = vdupq_n_u32(0xfffa3942);
const uint32x4_t T3_2  = vdupq_n_u32(0x8771f681);
const uint32x4_t T3_3  = vdupq_n_u32(0x6d9d6122);
const uint32x4_t T3_4  = vdupq_n_u32(0xfde5380c);
const uint32x4_t T3_5  = vdupq_n_u32(0xa4beea44);
const uint32x4_t T3_6  = vdupq_n_u32(0x4bdecfa9);
const uint32x4_t T3_7  = vdupq_n_u32(0xf6bb4b60);
const uint32x4_t T3_8  = vdupq_n_u32(0xbebfbc70);
const uint32x4_t T3_9  = vdupq_n_u32(0x289b7ec6);
const uint32x4_t T3_10 = vdupq_n_u32(0xeaa127fa);
const uint32x4_t T3_11 = vdupq_n_u32(0xd4ef3085);
const uint32x4_t T3_12 = vdupq_n_u32(0x04881d05);
const uint32x4_t T3_13 = vdupq_n_u32(0xd9d4d039);
const uint32x4_t T3_14 = vdupq_n_u32(0xe6db99e5);
const uint32x4_t T3_15 = vdupq_n_u32(0x1fa27cf8);
const uint32x4_t T3_16 = vdupq_n_u32(0xc4ac5665);

// T 常量（第四轮，Round 4）
const uint32x4_t T4_1  = vdupq_n_u32(0xf4292244);
const uint32x4_t T4_2  = vdupq_n_u32(0x432aff97);
const uint32x4_t T4_3  = vdupq_n_u32(0xab9423a7);
const uint32x4_t T4_4  = vdupq_n_u32(0xfc93a039);
const uint32x4_t T4_5  = vdupq_n_u32(0x655b59c3);
const uint32x4_t T4_6  = vdupq_n_u32(0x8f0ccc92);
const uint32x4_t T4_7  = vdupq_n_u32(0xffeff47d);
const uint32x4_t T4_8  = vdupq_n_u32(0x85845dd1);
const uint32x4_t T4_9  = vdupq_n_u32(0x6fa87e4f);
const uint32x4_t T4_10 = vdupq_n_u32(0xfe2ce6e0);
const uint32x4_t T4_11 = vdupq_n_u32(0xa3014314);
const uint32x4_t T4_12 = vdupq_n_u32(0x4e0811a1);
const uint32x4_t T4_13 = vdupq_n_u32(0xf7537e82);
const uint32x4_t T4_14 = vdupq_n_u32(0xbd3af235);
const uint32x4_t T4_15 = vdupq_n_u32(0x2ad7d2bb);
const uint32x4_t T4_16 = vdupq_n_u32(0xeb86d391);


/**
 * StringProcess: 将单个输入字符串转换成MD5计算所需的消息数组
 * @param input 输入
 * @param[out] n_byte 用于给调用者传递额外的返回值，即最终Byte数组的长度
 * @return Byte消息数组
 */
Byte *StringProcess(string input, int *n_byte)
{
	// 将输入的字符串转换为Byte为单位的数组
	Byte *blocks = (Byte *)input.c_str();
	int length = input.length();

	// 计算原始消息长度（以比特为单位）
	int bitLength = length * 8;

	// paddingBits: 原始消息需要的padding长度（以bit为单位）
	// 对于给定的消息，将其补齐至length%512==448为止
	// 需要注意的是，即便给定的消息满足length%512==448，也需要再pad 512bits
	int paddingBits = bitLength % 512;
	if (paddingBits > 448)
	{
		paddingBits = 512 - (paddingBits - 448);
	}
	else if (paddingBits < 448)
	{
		paddingBits = 448 - paddingBits;
	}
	else if (paddingBits == 448)
	{
		paddingBits = 512;
	}

	// 原始消息需要的padding长度（以Byte为单位）
	int paddingBytes = paddingBits / 8;
	// 创建最终的字节数组
	// length + paddingBytes + 8:
	// 1. length为原始消息的长度（bits）
	// 2. paddingBytes为原始消息需要的padding长度（Bytes）
	// 3. 在pad到length%512==448之后，需要额外附加64bits的原始消息长度，即8个bytes
	int paddedLength = length + paddingBytes + 8;
	Byte *paddedMessage = new Byte[paddedLength];

	// 复制原始消息
	memcpy(paddedMessage, blocks, length);

	// 添加填充字节。填充时，第一位为1，后面的所有位均为0。
	// 所以第一个byte是0x80
	paddedMessage[length] = 0x80;							 // 添加一个0x80字节
	memset(paddedMessage + length + 1, 0, paddingBytes - 1); // 填充0字节

	// 添加消息长度（64比特，小端格式）
	for (int i = 0; i < 8; ++i)
	{
		// 特别注意此处应当将bitLength转换为uint64_t
		// 这里的length是原始消息的长度
		paddedMessage[length + paddingBytes + i] = ((uint64_t)length * 8 >> (i * 8)) & 0xFF;
	}

	// 验证长度是否满足要求。此时长度应当是512bit的倍数
	int residual = 8 * paddedLength % 512;
	// assert(residual == 0);

	// 在填充+添加长度之后，消息被分为n_blocks个512bit的部分
	*n_byte = paddedLength;//paddedlength是paddedMessage的总字节数
	return paddedMessage;
}


/**
 * MD5Hash: 将单个输入字符串转换成MD5
 * @param input 输入
 * @param[out] state 用于给调用者传递额外的返回值，即最终的缓冲区，也就是MD5的结果
 * @return Byte消息数组
 */
void MD5Hash(string input[4], bit32 state[4][4])
{

	Byte *paddedMessage[4];
	int *messageLength = new int[4];
	for (int i = 0; i < 4; i += 1)
	{
		paddedMessage[i] = StringProcess(input[i], &messageLength[i]);
		// cout<<messageLength[i]<<endl;
		assert(messageLength[i] == messageLength[0]);
	}
	int n_blocks = messageLength[0] / 64;//messagelenth是总字节数，是512bit的倍数，也即64byte的倍数，所以messagelength/64即消息分为512bit的块数

	// bit32* state= new bit32[4];
	//对state初始化,分为4个32bit的部分,也即我们的state最终是32*4=128bits
	state[0][0] = 0x67452301;
	state[0][1] = 0xefcdab89;
	state[0][2] = 0x98badcfe;
	state[0][3] = 0x10325476;

	state[1][0] = 0x67452301;
	state[1][1] = 0xefcdab89;
	state[1][2] = 0x98badcfe;
	state[1][3] = 0x10325476;

	state[2][0] = 0x67452301;
	state[2][1] = 0xefcdab89;
	state[2][2] = 0x98badcfe;
	state[2][3] = 0x10325476;

	state[3][0] = 0x67452301;
	state[3][1] = 0xefcdab89;
	state[3][2] = 0x98badcfe;
	state[3][3] = 0x10325476;


	// 逐block地更新state(每个block是512bits)
	for (int i = 0; i < n_blocks; i += 1)
	{
		// bit32 x[16];//把每个512bits的block分为16个32bits的部分(小端整数)
		// x[16] 是准备好的 16 个 SIMD 向量，每个向量包含4个字符串的同一位置的 uint32_t
		uint32x4_t x[16];
		
		// 下面的处理，在理解上较为复杂
		for (int j = 0; j < 16; ++j)
		{	
			int base = i * 64 + j * 4;
			uint32_t word[4];  // 用来暂存每个字符串的第 j 个 uint32_t

			for (int z = 0; z < 4; ++z) {
				// 每个字符串的 paddedMessage 的第 j 个 32bit
				// 注意 MD5 使用 little-endian，需要组装 4 个 byte 成 uint32_t
				word[z] =
					((uint32_t)paddedMessage[z][base]) |
					((uint32_t)paddedMessage[z][base + 1] << 8) |
					((uint32_t)paddedMessage[z][base + 2] << 16) |
					((uint32_t)paddedMessage[z][base + 3] << 24);
			}
			// 创建 SIMD 向量
			x[j] = vld1q_u32(word);
		}
		//把字节转换为小端的 32 位字后，就能直接在后续的 FF、GG、HH、II 宏里，以 x[0]…x[15] 的形式去调用，跟算法描述一一对应、也方便做位运算。

		uint32x4_t a, b, c, d;
		a = (uint32x4_t){ state[0][0], state[1][0], state[2][0], state[3][0] };
		b = (uint32x4_t){ state[0][1], state[1][1], state[2][1], state[3][1] };
		c = (uint32x4_t){ state[0][2], state[1][2], state[2][2], state[3][2] };
		d = (uint32x4_t){ state[0][3], state[1][3], state[2][3], state[3][3] };


		auto start = system_clock::now();
		/* Round 1 */
		FF(a, b, c, d, x[0],  s11, T1_1);
		FF(d, a, b, c, x[1],  s12, T1_2);
		FF(c, d, a, b, x[2],  s13, T1_3);
		FF(b, c, d, a, x[3],  s14, T1_4);
		FF(a, b, c, d, x[4],  s11, T1_5);
		FF(d, a, b, c, x[5],  s12, T1_6);
		FF(c, d, a, b, x[6],  s13, T1_7);
		FF(b, c, d, a, x[7],  s14, T1_8);
		FF(a, b, c, d, x[8],  s11, T1_9);
		FF(d, a, b, c, x[9],  s12, T1_10);
		FF(c, d, a, b, x[10], s13, T1_11);
		FF(b, c, d, a, x[11], s14, T1_12);
		FF(a, b, c, d, x[12], s11, T1_13);
		FF(d, a, b, c, x[13], s12, T1_14);
		FF(c, d, a, b, x[14], s13, T1_15);
		FF(b, c, d, a, x[15], s14, T1_16);


		/* Round 2 */
		GG(a, b, c, d, x[1],  s21, T2_1);
		GG(d, a, b, c, x[6],  s22, T2_2);
		GG(c, d, a, b, x[11], s23, T2_3);
		GG(b, c, d, a, x[0],  s24, T2_4);
		GG(a, b, c, d, x[5],  s21, T2_5);
		GG(d, a, b, c, x[10], s22, T2_6);
		GG(c, d, a, b, x[15], s23, T2_7);
		GG(b, c, d, a, x[4],  s24, T2_8);
		GG(a, b, c, d, x[9],  s21, T2_9);
		GG(d, a, b, c, x[14], s22, T2_10);
		GG(c, d, a, b, x[3],  s23, T2_11);
		GG(b, c, d, a, x[8],  s24, T2_12);
		GG(a, b, c, d, x[13], s21, T2_13);
		GG(d, a, b, c, x[2],  s22, T2_14);
		GG(c, d, a, b, x[7],  s23, T2_15);
		GG(b, c, d, a, x[12], s24, T2_16);


		/* Round 3 */
		HH(a, b, c, d, x[5],  s31, T3_1);
		HH(d, a, b, c, x[8],  s32, T3_2);
		HH(c, d, a, b, x[11], s33, T3_3);
		HH(b, c, d, a, x[14], s34, T3_4);
		HH(a, b, c, d, x[1],  s31, T3_5);
		HH(d, a, b, c, x[4],  s32, T3_6);
		HH(c, d, a, b, x[7],  s33, T3_7);
		HH(b, c, d, a, x[10], s34, T3_8);
		HH(a, b, c, d, x[13], s31, T3_9);
		HH(d, a, b, c, x[0],  s32, T3_10);
		HH(c, d, a, b, x[3],  s33, T3_11);
		HH(b, c, d, a, x[6],  s34, T3_12);
		HH(a, b, c, d, x[9],  s31, T3_13);
		HH(d, a, b, c, x[12], s32, T3_14);
		HH(c, d, a, b, x[15], s33, T3_15);
		HH(b, c, d, a, x[2],  s34, T3_16);


		/* Round 4 */
		II(a, b, c, d, x[0],  s41, T4_1);
		II(d, a, b, c, x[7],  s42, T4_2);
		II(c, d, a, b, x[14], s43, T4_3);
		II(b, c, d, a, x[5],  s44, T4_4);
		II(a, b, c, d, x[12], s41, T4_5);
		II(d, a, b, c, x[3],  s42, T4_6);
		II(c, d, a, b, x[10], s43, T4_7);
		II(b, c, d, a, x[1],  s44, T4_8);
		II(a, b, c, d, x[8],  s41, T4_9);
		II(d, a, b, c, x[15], s42, T4_10);
		II(c, d, a, b, x[6],  s43, T4_11);
		II(b, c, d, a, x[13], s44, T4_12);
		II(a, b, c, d, x[4],  s41, T4_13);
		II(d, a, b, c, x[11], s42, T4_14);
		II(c, d, a, b, x[2],  s43, T4_15);
		II(b, c, d, a, x[9],  s44, T4_16);


		// 先手动聚合成向量
		uint32_t col0[4] = { state[0][0], state[1][0], state[2][0], state[3][0] };
		uint32_t col1[4] = { state[0][1], state[1][1], state[2][1], state[3][1] };
		uint32_t col2[4] = { state[0][2], state[1][2], state[2][2], state[3][2] };
		uint32_t col3[4] = { state[0][3], state[1][3], state[2][3], state[3][3] };

		uint32x4_t s0 = vld1q_u32(col0);
		uint32x4_t s1 = vld1q_u32(col1);
		uint32x4_t s2 = vld1q_u32(col2);
		uint32x4_t s3 = vld1q_u32(col3);

		// SIMD 累加
		s0 = vaddq_u32(s0, a);
		s1 = vaddq_u32(s1, b);
		s2 = vaddq_u32(s2, c);
		s3 = vaddq_u32(s3, d);

		// 手动分散回 state
		vst1q_u32(col0, s0);
		vst1q_u32(col1, s1);
		vst1q_u32(col2, s2);
		vst1q_u32(col3, s3);

		for (int k = 0; k < 4; ++k) {
			state[k][0] = col0[k];
			state[k][1] = col1[k];
			state[k][2] = col2[k];
			state[k][3] = col3[k];
		}
		//最终的 state[i] 就是第 i 个字符串的 128-bit MD5 哈希（用 4 个 32-bit 表示）。
	}

	for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            // 小端字节顺序转换为大端字节顺序
			uint32_t value = state[i][j];
            state[i][j] = 
				((value & 0xFF) << 24) |
                ((value & 0xFF00) << 8) |
                ((value & 0xFF0000) >> 8) |
                ((value & 0xFF000000) >> 24);

            // // 输出格式为8位十六进制，保证每个部分8位，不足补零
            // ss << setw(8) << setfill('0') << hex << v;
        }
    }
	//假设某次计算后 state[0] = 0xA1B2C3D4(在小端内存里存作 D4 C3 B2 A1),合成后 state[0] = 0xD4C3B2A1，其内存顺序变成 A1 B2 C3 D4，与大端序输出一致。
	//这样，内存中按 state[0] 的自然排列，即是正确的 MD5 字节序。
	
	// 输出最终的hash结果
	// for (int j = 0; j < 4; j += 1)
	// {
	// 	cout << std::setw(8) << std::setfill('0') << hex << state[j];
	// }
	// cout << endl;

	// 释放动态分配的内存
	// 实现SIMD并行算法的时候，也请记得及时回收内存！
	for(int i=0;i<4;i++) delete[] paddedMessage[i];
	delete[] messageLength;
}

