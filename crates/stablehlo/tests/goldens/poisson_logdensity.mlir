module attributes {flatppl.runtime_status = 1 : i64} {
  func.func @logdensity(%arg0: tensor<f32>) -> (tensor<f32>, tensor<i32>) {
    %0 = stablehlo.constant dense<0.0> : tensor<f32>
    %1 = stablehlo.compare LT, %arg0, %0 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %2 = stablehlo.constant dense<0> : tensor<i32>
    %3 = stablehlo.constant dense<9> : tensor<i32>
    %4 = stablehlo.select %1, %3, %2 : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %6 = stablehlo.constant dense<"0x00004040"> : tensor<f32>
    %8 = stablehlo.constant dense<1.0> : tensor<f32>
    %9 = stablehlo.compare EQ, %arg0, %0 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %10 = stablehlo.select %9, %8, %arg0 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %15 = stablehlo.divide %10, %8 : tensor<f32>
    %16 = stablehlo.compare GT, %15, %0 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %17 = stablehlo.select %16, %15, %10 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %18 = stablehlo.log %17 : tensor<f32>
    %22 = stablehlo.multiply %6, %18 : tensor<f32>
    %24 = stablehlo.constant dense<"0x6058E53F"> : tensor<f32>
    %25 = stablehlo.subtract %22, %10 : tensor<f32>
    %26 = stablehlo.subtract %25, %24 : tensor<f32>
    %70 = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %71 = stablehlo.negate %70 : tensor<f32>
    %72 = stablehlo.select %9, %71, %26 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    return %72, %4 : tensor<f32>, tensor<i32>
  }
}
