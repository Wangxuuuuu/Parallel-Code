#include<iostream>
#include<windows.h>
using namespace std;
int main(){
    int N=10;int step=10;
    for(int N=10;N<=1000;N+=step){
        if(N==100) step=100;
        int counter=0;
        double elapsedmseconds=0;;
        while(counter<500){//ÿ��N�ظ�500�μ�ʱ����ƽ��ֵ���ﵽ��׼��ʱ
            counter++;
            int** M=new int*[N]{};
            for(int i=0;i<N;i++){
                M[i]=new int[N]{};
            }
            for(int i=0;i<N;i++){
                for(int j=0;j<N;j++){
                    M[i][j]=i+j;
                }
            }//����þ���M[N][N]
            int* a=new int[N];
            for(int i=0;i<N;i++){
                a[i]=i;
            }//�������˵�����a
            int* sum=new int[N]{};//��ʼ������sum�ڻ�����

            LARGE_INTEGER frequency;
            QueryPerformanceFrequency(&frequency);
            LARGE_INTEGER startcount;
            QueryPerformanceCounter(&startcount);

            for(int j=0;j<N;j++){
                for(int i=0;i<N;i++){
                    sum[j]+=M[i][j]*a[i];//ֱ�Ӳ�ȡƽ���㷨
                }
            }

            LARGE_INTEGER endcount;
            QueryPerformanceCounter(&endcount);
            elapsedmseconds+=1000*(double)(endcount.QuadPart-startcount.QuadPart)/(double)frequency.QuadPart;
        }
        cout<<"N: "<<N<<" ʱ��: "<<elapsedmseconds/counter<<" ms "<<endl;
    }
    return 0;
}

// #include<iostream>
// using namespace std;
// int main(){
//     clock_t startcount,endcount;
//     int N=10;int step=10;
//     long counter=0;
//     for(int N=10;N<=1000;N+=step){
//     if(N==100) step=100;
//     startcount=clock(); counter=0;
//     while((clock()-startcount)/(double)CLOCKS_PER_SEC<0.1){//��ʱ��<0.1s
//         counter++;
//     int** M=new int*[N]{};
//     for(int i=0;i<N;i++){
//         M[i]=new int[N]{};
//     }
//     for(int i=0;i<N;i++){
//         for(int j=0;j<N;j++){
//             M[i][j]=i+j;
//         }
//     }//����þ���M[N][N]
//     int* a=new int[N];
//     for(int i=0;i<N;i++){
//         a[i]=i;
//     }//�������˵�����a
//     int* sum=new int[N]{};//��ʼ������sum�ڻ�����

    
//     for(int j=0;j<N;j++){
//         for(int i=0;i<N;i++){
//             sum[j]+=M[i][j]*a[i];//ֱ�Ӳ�ȡƽ���㷨
//         }
//     }
// }

//     endcount=clock();
//     double micorseconds=1000*(double)(endcount-startcount)/double(CLOCKS_PER_SEC);
//     cout<<"N: "<<N<<" ѭ������ "<<counter<<" ��ʱ�� "<<micorseconds<<" ms "<<" ƽ��ÿ��ʱ�� "<<micorseconds/counter<<" ms "<<endl;
// }
//     return 0;
// }