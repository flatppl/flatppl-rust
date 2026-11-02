module attributes {flatppl.runtime_status = 1 : i64} {
  func.func @logdensity(%arg0: tensor<f32>, %arg1: tensor<f32>) -> (tensor<f32>, tensor<i32>) {
    %0 = stablehlo.constant dense<0.0> : tensor<f32>
    %1 = stablehlo.compare LE, %arg0, %0 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %2 = stablehlo.compare LE, %arg1, %0 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %3 = stablehlo.select %1, %1, %2 : (tensor<i1>, tensor<i1>, tensor<i1>) -> tensor<i1>
    %4 = stablehlo.constant dense<0> : tensor<i32>
    %5 = stablehlo.constant dense<9> : tensor<i32>
    %6 = stablehlo.select %3, %5, %4 : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %8 = stablehlo.constant dense<"0x00000040"> : tensor<f32>
    %9 = stablehlo.add %8, %arg1 : tensor<f32>
    %10 = chlo.lgamma %9 : tensor<f32> -> tensor<f32>
    %11 = chlo.lgamma %arg1 : tensor<f32> -> tensor<f32>
    %12 = stablehlo.negate %11 : tensor<f32>
    %16 = stablehlo.constant dense<"0x187231BF"> : tensor<f32>
    %17 = stablehlo.add %10, %12 : tensor<f32>
    %18 = stablehlo.add %17, %16 : tensor<f32>
    %19 = stablehlo.add %arg0, %arg1 : tensor<f32>
    %20 = stablehlo.log %19 : tensor<f32>
    %21 = stablehlo.negate %20 : tensor<f32>
    %22 = stablehlo.log %arg0 : tensor<f32>
    %23 = stablehlo.add %22, %21 : tensor<f32>
    %24 = stablehlo.multiply %8, %23 : tensor<f32>
    %25 = stablehlo.log %arg1 : tensor<f32>
    %26 = stablehlo.add %25, %21 : tensor<f32>
    %27 = stablehlo.multiply %arg1, %26 : tensor<f32>
    %28 = stablehlo.add %18, %24 : tensor<f32>
    %29 = stablehlo.add %28, %27 : tensor<f32>
    return %29, %6 : tensor<f32>, tensor<i32>
  }
}
