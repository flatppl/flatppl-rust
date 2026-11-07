module attributes {flatppl.runtime_status = 1 : i64} {
  func.func @logdensity(%arg0: tensor<f32>, %arg1: tensor<f32>) -> (tensor<f32>, tensor<i32>) {
    %0 = stablehlo.minimum %arg0, %arg1 : tensor<f32>
    %1 = stablehlo.constant dense<0.0> : tensor<f32>
    %2 = stablehlo.compare LE, %0, %1 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %3 = stablehlo.constant dense<0> : tensor<i32>
    %4 = stablehlo.constant dense<9> : tensor<i32>
    %5 = stablehlo.select %2, %4, %3 : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %7 = stablehlo.constant dense<1.0> : tensor<f32>
    %9 = stablehlo.subtract %arg0, %7 : tensor<f32>
    %10 = stablehlo.constant dense<"0x187231BF"> : tensor<f32>
    %11 = stablehlo.multiply %9, %10 : tensor<f32>
    %12 = stablehlo.subtract %arg1, %7 : tensor<f32>
    %14 = stablehlo.multiply %12, %10 : tensor<f32>
    %15 = chlo.lgamma %arg0 : tensor<f32> -> tensor<f32>
    %16 = stablehlo.negate %15 : tensor<f32>
    %17 = chlo.lgamma %arg1 : tensor<f32> -> tensor<f32>
    %18 = stablehlo.negate %17 : tensor<f32>
    %19 = stablehlo.add %arg0, %arg1 : tensor<f32>
    %20 = chlo.lgamma %19 : tensor<f32> -> tensor<f32>
    %21 = stablehlo.add %11, %14 : tensor<f32>
    %22 = stablehlo.add %16, %18 : tensor<f32>
    %23 = stablehlo.add %22, %20 : tensor<f32>
    %24 = stablehlo.add %21, %23 : tensor<f32>
    return %24, %5 : tensor<f32>, tensor<i32>
  }
}
