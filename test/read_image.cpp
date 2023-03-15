/*
 * @Author: xzj
 * @Date: 2023-03-14 14:18:33
 * @LastEditTime: 2023-03-15 03:01:35
 * @Description: 
 * @FilePath: /scnni/readimg.cpp
 */
#include <fstream> // ifstream, ifstream::in
#include <iostream>
using std::cout;
using std::endl;

TEST(){
	// 1. 打开图片文件
	std::ifstream is("test.jpg", std::ifstream::in | std::ios::binary);
	// 2. 计算图片长度
	is.seekg(0, std::ifstream::end);  //将文件流指针定位到流的末尾
	int length = is.tellg();
	is.seekg(0, std::ifstream::beg);  //将文件流指针重新定位到流的开始
	// 3. 创建内存缓存区
	char * buffer = new char[length];
	// 4. 读取图片
	is.read(buffer, length);
	// 到此，图片已经成功的被读取到内存（buffer）中
	delete [] buffer;
	return 0;
}