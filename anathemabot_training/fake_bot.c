#include "fake_bot.h"

#include <assert.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#include <signal.h>

int closeFlag = 0;

void term(int signum) {
    fprintf(stderr, "hrerekrejl");
    closeFlag = 1;
}



int main(int argc, char * argv[]) {

    struct sigaction action;
    memset(&action, 0, sizeof(action));
    action.sa_handler = term;
    sigaction(SIGTERM, &action, NULL);
    
    if (argc != 2) {
        fprintf(stdout, "fake_bot did not get fifo ID!");
        exit(EXIT_FAILURE);
    }

    int fifoID = atoi(argv[1]);

    char toFifoName[strlen(TO_HALITE_PREFIX) + 2];
    char fromFifoName[strlen(FROM_HALITE_PREFIX) + 2];

    snprintf(toFifoName, strlen(TO_HALITE_PREFIX) + 2, "%s%d", TO_HALITE_PREFIX, fifoID);
    snprintf(fromFifoName, strlen(FROM_HALITE_PREFIX) + 2, "%s%d", FROM_HALITE_PREFIX, fifoID);

    // open named pipes
    int fromPipeFd = open(fromFifoName, O_WRONLY);
    int toPipeFd = open(toFifoName, O_RDONLY);

    if (fromPipeFd < 0) {
        perror("from pipe open");
        exit(EXIT_FAILURE);
    }

    if (toPipeFd < 0) {
        perror("to pipe open");
        exit(EXIT_FAILURE);
    }

    // STDIN -> from_pipe fd
    // to_pipe fd -> STDOUT

    // int pipe[2];

    // if (pipe(pipe) < 0) {
    //     perror("Pipe creation failed.");
    //     exit(EXIT_FAILURE);
    // }
    
    //dup2(STDIN_FILENO, pipe[0]);
    int n;
    char buff[1000];
    //int log;
    switch (fork()) {
        case -1:
            perror("Fork failed.");
            exit(EXIT_FAILURE);
        case 0: // Child
            ;
            //char logName[1000];

            //snprintf(logName, 11, "Child%d.log", fifoID);

            //log = open(logName, O_WRONLY|O_CREAT, 0666);

            close(toPipeFd);
            // close(pipe[0]);

            while ((n = read(STDIN_FILENO, buff, 1000)) > 0) {

                if (closeFlag) {
                    //write(log, "earera.\n", 7);
                    printf("here");
                    close(fromPipeFd);
                    exit(EXIT_SUCCESS);
                }

                //int other = write(log, buff, n);
                int bytesWritten = write(fromPipeFd, buff, n);
            }

            

            int bytes = write(fromPipeFd, "Done.\n", 6);
            //write(log, "Done.\n", 6);

            close(fromPipeFd);
            //close(log);
            break;
        default: // Parent
            //log = open("parent.log", O_WRONLY|O_CREAT, 0666);

            close(fromPipeFd);
            // close(pipe[0]);
            // close(pipe[1]);

            while ((n = read(toPipeFd, buff, 1000)) > 0) {
                //int other = write(log, buff, n);
                int bytesWritten = write(STDOUT_FILENO, buff, n);
            }
            close(toPipeFd);
    }

    
    

    // char buff[1000];

    // while (read(STDIN_FILENO, buff, 1000) > 0) {
        
    //     int bytesWritten = write(fromPipeFd, buff, 1000);
        

    // }

    return EXIT_SUCCESS;

}