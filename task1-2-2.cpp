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
        while(counter<500){//ÿ��N�ظ�500�μ�ʱ����ƽ��ֵ���ﵽ��׼��ʱ
            counter++;
            int* a=new int[N]{};
            for(int i=0;i<N;i++){
                a[i]=i;
            }//�����Ԫ������a
            int sum=0;//���Ϊsum

            LARGE_INTEGER frequency;
            QueryPerformanceFrequency(&frequency);
            LARGE_INTEGER startcount;
            QueryPerformanceCounter(&startcount);

            for(int m=N;m>1;m/=2){
                for(int i=0;i<m/2;i++){
                    a[i]=a[2*i]+a[2*i+1];
                }
            }//�����Ż��㷨����ѭ��ʵ�ֵݹ�
            sum=a[0];//a[0]Ϊ���ս��

            LARGE_INTEGER endcount;
            QueryPerformanceCounter(&endcount);
            elapsedmseconds+=1000*(double)(endcount.QuadPart-startcount.QuadPart)/(double)frequency.QuadPart;
        }
        cout<<"N: "<<N<<" ʱ��: "<<elapsedmseconds/counter<<" ms "<<endl;
    }
    return 0;
}