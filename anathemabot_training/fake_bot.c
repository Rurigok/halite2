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

int main(int argc, char * argv[]) {

    int fifoID = atoi(argv[1]);

    char toFifoName[strlen(TO_HALITE_PREFIX) + 2];
    char fromFifoName[strlen(FROM_HALITE_PREFIX) + 2];

    snprintf(toFifoName, strlen(TO_HALITE_PREFIX) + 2, "%s%d", TO_HALITE_PREFIX, fifoID);
    snprintf(fromFifoName, strlen(FROM_HALITE_PREFIX) + 2, "%s%d", FROM_HALITE_PREFIX, fifoID);

    // open named pipes
    int fromPipeFd = open(fromFifoName, O_RDONLY);
    int toPipeFd = open(toFifoName, O_WRONLY);

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

    char buff[1000];

    switch (fork()) {
        case -1:
            perror("Fork failed.");
            exit(EXIT_FAILURE);
        case 0: // Child
            

            close(toPipeFd);
            // close(pipe[0]);

            while (read(STDIN_FILENO, buff, 1000) > 0) {
                int bytesWritten = write(fromPipeFd, buff, 1000);
            }
            break;
        default: // Parent
            

            close(fromPipeFd);
            // close(pipe[0]);
            // close(pipe[1]);

            while (read(toPipeFd, buff, 1000) > 0) {
                int bytesWritten = write(STDOUT_FILENO, buff, 1000);
            }
    }


    // char buff[1000];

    // while (read(STDIN_FILENO, buff, 1000) > 0) {
        
    //     int bytesWritten = write(fromPipeFd, buff, 1000);
        

    // }

    return EXIT_SUCCESS;

}