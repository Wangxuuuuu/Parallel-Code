#include<iostream>
#include<windows.h>
#include<cmath>
using namespace std;
int main(){
    int mi=10;
    int step=1;
    for(mi=10;mi<=20;mi+=step){
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

            for(int i=0;i<N;i++){
                sum+=a[i];
            }//直接采用平凡算法链式相加

            LARGE_INTEGER endcount;
            QueryPerformanceCounter(&endcount);
            elapsedmseconds+=1000*(double)(endcount.QuadPart-startcount.QuadPart)/(double)frequency.QuadPart;
        }
        cout<<"N: "<<N<<" 时间: "<<elapsedmseconds/counter<<" ms "<<endl;
    }
    return 0;
}