#include<iostream>
#include<windows.h>
#include<cmath>
using namespace std;
int main(){
    int mi=1;
    int step=1;
    for(mi=1;mi<=20;mi+=step){
        // if(mi==10) step=1;
        int N=pow(2,mi);
        int counter=0;
        double elapsedmseconds=0;
        while(counter<500){//每个N重复500次计时，求平均值，达到精准计时
            counter++;
            int* a=new int[N]{};
            for(int i=0;i<N;i++){
                a[i]=i;
            }//定义好元素数组a
            int sum=0;//结果为sum

            LARGE_INTEGER frequency;
            QueryPerformanceFrequency(&frequency);
            LARGE_INTEGER startcount;
            QueryPerformanceCounter(&startcount);

            for(int m=N;m>1;m/=2){
                for(int i=0;i<m/2;i++){
                    a[i]=a[2*i]+a[2*i+1];
                }
            }//采用优化算法二重循环实现递归
            sum=a[0];//a[0]为最终结果

            LARGE_INTEGER endcount;
            QueryPerformanceCounter(&endcount);
            elapsedmseconds+=1000*(double)(endcount.QuadPart-startcount.QuadPart)/(double)frequency.QuadPart;
        }
        cout<<"N: "<<N<<" 时间: "<<elapsedmseconds/counter<<" ms "<<endl;
    }
    return 0;
}
