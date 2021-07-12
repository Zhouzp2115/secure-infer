set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../libs)

set (CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -O3")

#lib dir
link_directories("../libs")

#libgc
add_library(gc SHARED "gc/gc.cpp")
target_link_libraries(gc ${RELIC_LIBRARIES}   
                         ${EMP-OT_LIBRARIES}  ${OPENSSL_LIBRARIES}  
                         ${Boost_LIBRARIES}   ${GMP_LIBRARIES} ${TORCH_LIBRARIES})

#lib ppml
file(GLOB ppmlsrc "base/*.cpp" "utils/*.cpp")
add_library(ppml SHARED ${ppmlsrc})
target_link_libraries(ppml  ${TORCH_LIBRARIES} "gc")

#infer dameo
add_executable(test_func "infer/test_func.cpp")
target_link_libraries(test_func "gc" "ppml")

#NetworkA-MNIST
add_executable(NetworkA_MNIST "infer/NetworkA-MNIST.cpp")
target_link_libraries(NetworkA_MNIST "gc" "ppml")

#NetworkA-FASHIONMNIST
add_executable(NetworkA_FASHIONMNIST "infer/NetworkA-FASHIONMNIST.cpp")
target_link_libraries(NetworkA_FASHIONMNIST "gc" "ppml")

#NetworkB-MNIST
add_executable(NetworkB_MNIST "infer/NetworkB-MNIST.cpp")
target_link_libraries(NetworkB_MNIST "gc" "ppml")

#NetworkB-FASHIONMNIST
add_executable(NetworkB_FASHIONMNIST "infer/NetworkB-FASHIONMNIST.cpp")
target_link_libraries(NetworkB_FASHIONMNIST "gc" "ppml")

#NetworkC-MNIST
add_executable(NetworkC_MNIST "infer/NetworkC-MNIST.cpp")
target_link_libraries(NetworkC_MNIST "gc" "ppml")

#NetworkC-FASHIONMNIST
add_executable(NetworkC_FASHIONMNIST "infer/NetworkC-FASHIONMNIST.cpp")
target_link_libraries(NetworkC_FASHIONMNIST "gc" "ppml")

#LeNet-MNIST
add_executable(LeNet_MNIST "infer/LeNet-MNIST.cpp")
target_link_libraries(LeNet_MNIST "gc" "ppml")

#LeNet-FASHIONMNIST
add_executable(LeNet_FASHIONMNIST "infer/LeNet-FASHIONMNIST.cpp")
target_link_libraries(LeNet_FASHIONMNIST "gc" "ppml")

#LeNet-CIFAR10
add_executable(LeNet_CIFAR10 "infer/LeNet-CIFAR10.cpp")
target_link_libraries(LeNet_CIFAR10 "gc" "ppml")

