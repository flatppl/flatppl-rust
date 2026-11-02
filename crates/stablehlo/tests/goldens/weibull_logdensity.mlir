module attributes {flatppl.runtime_status = 1 : i64} {
  func.func @logdensity(%arg0: tensor<f32>, %arg1: tensor<f32>) -> (tensor<f32>, tensor<i32>) {
    %0 = stablehlo.minimum %arg0, %arg1 : tensor<f32>
    %1 = stablehlo.constant dense<0.0> : tensor<f32>
    %2 = stablehlo.compare LE, %0, %1 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %3 = stablehlo.constant dense<0> : tensor<i32>
    %4 = stablehlo.constant dense<9> : tensor<i32>
    %5 = stablehlo.select %2, %4, %3 : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %6 = stablehlo.constant dense<0.5> : tensor<f32>
    %8 = stablehlo.constant dense<1.0> : tensor<f32>
    %9 = stablehlo.log %arg0 : tensor<f32>
    %10 = stablehlo.log %arg1 : tensor<f32>
    %11 = stablehlo.negate %10 : tensor<f32>
    %12 = stablehlo.divide %6, %arg1 : tensor<f32>
    %13 = stablehlo.log %12 : tensor<f32>
    %14 = stablehlo.subtract %arg0, %8 : tensor<f32>
    %15 = stablehlo.multiply %14, %13 : tensor<f32>
    %16 = stablehlo.power %12, %arg0 : tensor<f32>
    %17 = stablehlo.negate %16 : tensor<f32>
    %18 = stablehlo.add %9, %11 : tensor<f32>
    %19 = stablehlo.add %18, %15 : tensor<f32>
    %20 = stablehlo.add %19, %17 : tensor<f32>
    return %20, %5 : tensor<f32>, tensor<i32>
  }
}
