#include <thread>
#include <iostream>

void do_work(int person_id)
{
    std::cout << "Some random person's ID: " << person_id << "\n";
}

int main()
{
    for (int i{0}; i < 100; ++i)
    {
        std::thread t1(do_work, 1); // creating thread 1 which takes in a function pointer and some forward param
        std::thread t2(do_work, 2); // creating thread 2 \/\/\/\/\/...

        t1.join();
        t2.join();
    }

    return 0;
}