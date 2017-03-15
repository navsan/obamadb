
#ifndef OBAMADB_LOGUTILS_H_
#define OBAMADB_LOGUTILS_H_

#define VPRINT(str) { if(FLAGS_verbose) { printf(str); } }
#define VPRINTF(str, ...) { if(FLAGS_verbose) { printf(str, __VA_ARGS__); } }
#define VSTREAM(obj) {if(FLAGS_verbose){ std::cout << obj <<std::endl; }}




#endif // #ifndef OBAMADB_LOGUTILS_H_
