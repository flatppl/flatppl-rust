module attributes {flatppl.runtime_status = 1 : i64} {
  func.func @logdensity(%arg0: tensor<f32>, %arg1: tensor<f32>) -> (tensor<f32>, tensor<i32>) {
    %0 = stablehlo.constant dense<0.0> : tensor<f32>
    %1 = stablehlo.compare LE, %arg1, %0 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %2 = stablehlo.constant dense<0> : tensor<i32>
    %3 = stablehlo.constant dense<9> : tensor<i32>
    %4 = stablehlo.select %1, %3, %2 : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %5 = stablehlo.constant dense<0.5> : tensor<f32>
    %6 = stablehlo.constant dense<-1.1447298858494002> : tensor<f32>
    %7 = stablehlo.log %arg1 : tensor<f32>
    %8 = stablehlo.negate %7 : tensor<f32>
    %9 = stablehlo.subtract %5, %arg0 : tensor<f32>
    %10 = stablehlo.divide %9, %arg1 : tensor<f32>
    %11 = stablehlo.multiply %10, %10 : tensor<f32>
    %12 = stablehlo.constant dense<1.0> : tensor<f32>
    %13 = stablehlo.add %12, %11 : tensor<f32>
    %14 = stablehlo.log %13 : tensor<f32>
    %15 = stablehlo.negate %14 : tensor<f32>
    %16 = stablehlo.add %6, %8 : tensor<f32>
    %17 = stablehlo.add %16, %15 : tensor<f32>
    return %17, %4 : tensor<f32>, tensor<i32>
  }
}
