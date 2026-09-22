module {
  func.func @sample(%key: tensor<2xui64>) -> (tensor<f32>, tensor<2xui64>) {
    %0 = stablehlo.constant dense<0.0> : tensor<f32>
    %1 = stablehlo.constant dense<1.0> : tensor<f32>
    %2, %3 = stablehlo.rng_bit_generator %key, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<ui32>)
    %4 = stablehlo.constant dense<9> : tensor<ui32>
    %5 = stablehlo.shift_right_logical %3, %4 : tensor<ui32>
    %6 = stablehlo.convert %5 : (tensor<ui32>) -> tensor<f32>
    %7 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
    %8 = stablehlo.multiply %6, %7 : tensor<f32>
    %9 = stablehlo.constant dense<0.5> : tensor<f32>
    %10 = stablehlo.subtract %8, %9 : tensor<f32>
    %11 = stablehlo.compare GE, %10, %0 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %12 = stablehlo.constant dense<-1.0> : tensor<f32>
    %13 = stablehlo.select %11, %1, %12 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %14 = stablehlo.abs %10 : tensor<f32>
    %15 = stablehlo.constant dense<2.0> : tensor<f32>
    %16 = stablehlo.multiply %15, %14 : tensor<f32>
    %17 = stablehlo.subtract %1, %16 : tensor<f32>
    %18 = stablehlo.log %17 : tensor<f32>
    %19 = stablehlo.multiply %13, %18 : tensor<f32>
    %20 = stablehlo.multiply %1, %19 : tensor<f32>
    %21 = stablehlo.negate %20 : tensor<f32>
    %22 = stablehlo.add %0, %21 : tensor<f32>
    return %22, %2 : tensor<f32>, tensor<2xui64>
  }
}
