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

    // generate a unique ID via current nanosecond time
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);

    const int n = snprintf(NULL, 0, "%ld", ts.tv_nsec);
    assert(n > 0);
    char timeId[n + 1];
    int c = snprintf(timeId, n + 1, "%lu", ts.tv_nsec);
    assert(timeId[n] == '\0');
    assert(c == n);

    int fifoNameSz = strlen(NAMED_PIPE_PREFIX) + n;
    char fifoName[fifoNameSz];
    memset(fifoName, 0, fifoNameSz);
    strcat(fifoName, NAMED_PIPE_PREFIX);
    strcat(fifoName, timeId);

    if (fifoName[strlen(NAMED_PIPE_PREFIX) + n] == '\0') {
        printf("end is null");
    }

    printf("%s\n", fifoName);

    // create named pipe
    if (mkfifo(fifoName, 0666) != 0) {
        perror("pipe");
        exit(EXIT_FAILURE);
    }

    // open named pipe
    int pipeFd = open(fifoName, O_RDWR);

    if (pipeFd < 0) {
        perror("pipe open");
        exit(EXIT_FAILURE);
    }

    // stdin to this program goes to input fd of named pipe
    if (dup2(pipeFd, STDIN_FILENO) < 0) {
        perror("redirect stdin");
        exit(EXIT_FAILURE);
    }

    // output fd of named pipe goes to stdout of this program
    if (dup2(pipeFd, STDOUT_FILENO) < 0) {
        perror("redirect stdout");
        exit(EXIT_FAILURE);
    }

    close(pipeFd);
    return EXIT_SUCCESS;

}
